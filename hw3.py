from openai import OpenAI
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from googlesearch import search
import ollama
import chromadb
from langchain_community.document_loaders import PyPDFDirectoryLoader  # Importing PDF loader from Langchain
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Importing text splitter from Langchain
from langchain.schema import Document  # Importing Document schema from Langchain
from chromadb.config import DEFAULT_TENANT, DEFAULT_DATABASE, Settings
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
from googlesearch import search
import pickle
import os.path
import configparser
import os
from getpass import getpass
import re


def load_documents(file_path):
    document_loader = PyPDFDirectoryLoader(file_path)  # Initialize PDF loader with specified directory
    return document_loader.load()  # Load PDF documents and return them as a list of Document objects

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,  # Size of each chunk in characters
        chunk_overlap=100,  # Overlap between consecutive chunks
        length_function=len,  # Function to compute the length of the text
        add_start_index=True,  # Flag to add start index to each chunk
    )
    # Split documents into smaller chunks using text splitter
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    # Print example of page content and metadata for a chunk
    document = chunks[10]
    print(document.page_content)
    print(document.metadata)

    return chunks  # Return the list of split text chunks

clientpdf = chromadb.PersistentClient(
    path="chromadb",
    settings=Settings(),
    tenant=DEFAULT_TENANT,
    database=DEFAULT_DATABASE,
)

client = OpenAI()


def authenticate_google_calendar():
    creds = None
    if os.path.exists('token.pickle'):
        with open('token.pickle', 'rb') as token:
            creds = pickle.load(token)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json',
                                                             scopes=['https://www.googleapis.com/auth/calendar'])
            creds = flow.run_local_server(port=0)
        with open('token.pickle', 'wb') as token:
            pickle.dump(creds, token)
    return build('calendar', 'v3', credentials=creds)

def create_event(summary, location, start_time, end_time, gc, attendees=None):
    event = {
        'summary': summary,
        'location': location,
        'start': {'dateTime': start_time, 'timeZone': 'America/Los_Angeles'},
        'end': {'dateTime': end_time, 'timeZone': 'America/Los_Angeles'},
        'attendees': [{'email': att} for att in attendees] if attendees else [],
    }
    event = gc.events().insert(calendarId='primary', body=event).execute()
    print('Event created: %s' % (event.get('htmlLink')))


class PersonalAIAssistant:
    def __init__(self, config_path='config.ini'):
        self.config_path = config_path
        self.config = configparser.ConfigParser()
        self.load_config()
        self.assistant = client.beta.assistants.create(
                name="Personal Assistant",
                instructions="You are a personal assistant to help with daily works.",
                tools=[{"type": "code_interpreter"}],
                model="gpt-4o",
            )
        self.main_thread = client.beta.threads.create()
        self.send_email_thread = client.beta.threads.create()
        self.schedule_meeting_thread = client.beta.threads.create()
        self.search_internet_thread = client.beta.threads.create()
        self.read_pdf_thread = client.beta.threads.create()

        # # Load and prompt for missing configuration directly as class attributes
        self.api_key = self.get_or_request_info('openai', 'api_key', "Enter your OpenAI API key: ", secret=True)
        self.email_address = self.get_or_request_info('email', 'address', "Enter your email address: ")
        self.email_password = self.get_or_request_info('email', 'password', "Enter your email password: ", secret=True)

    def load_config(self):
        if not os.path.exists(self.config_path):
            # Create the config file if it does not exist
            with open(self.config_path, 'w') as configfile:
                self.config.write(configfile)
        self.config.read(self.config_path)

    def get_or_request_info(self, section, option, prompt, secret=False):
        if self.config.has_option(section, option):
            return self.config.get(section, option)
        else:
            # Request missing configuration from the user and save it
            if secret:
                value = getpass(prompt)
            else:
                value = input(prompt)
            self.update_config(section, option, value)
            return value

    def update_config(self, section, key, value):
        if not self.config.has_section(section):
            self.config.add_section(section)
        self.config.set(section, key, value)
        with open(self.config_path, 'w') as configfile:
            self.config.write(configfile)

    def handle_query(self, response):
        # Use OpenAI's Assistant API to determine intent and extract necessary details
        intent = self.parse_intent(response)
        print(intent)
        if intent == "send_email":
            message = client.beta.threads.messages.create(
                thread_id=self.send_email_thread.id,
                role="user",
                content=f"Now the user will send an email, here's the email information: '{response}'\n\n .f"
            )
            run = client.beta.threads.runs.create_and_poll(
                thread_id=self.send_email_thread.id,
                assistant_id=self.assistant.id,
                instructions="Please extract and only return the formatted receipt email as the following format: **To:** \n **Subject:** Meeting Update \n **Body:** "
            )
            email_text = "**To:** John Doe **Subject:** Meeting Update **Body:** Please see the attached document for details."
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=self.send_email_thread.id
                )
                print("receipt", messages.data[0].content[0].text.value)
                email_text = messages.data[0].content[0].text.value
            to_match = re.search(r"\*\*To:\*\*\s*(.*)", email_text)
            subject_match = re.search(r"\*\*Subject:\*\*\s*(.*)", email_text)
            body_match = re.search(r"\*\*Body:\*\*\s*(.*)", email_text)

            # Extracted values
            to_email = to_match.group(1).strip() if to_match else None
            subject = subject_match.group(1).strip() if subject_match else None
            body = body_match.group(1).strip() if body_match else None

            # print(to_email, subject, body)

            return self.send_email(to_email, subject, body)
        elif intent == "schedule_meeting":
            message = client.beta.threads.messages.create(
                thread_id=self.schedule_meeting_thread.id,
                role="user",
                content=f"Now the user will schedule meeting on calendar. Here's the entered information from user: '{response}'\n\n .f"
            )
            run = client.beta.threads.runs.create_and_poll(
                thread_id=self.schedule_meeting_thread.id,
                assistant_id=self.assistant.id,
                instructions="Please extract the meeting information from users input and only return the formatted response as the following format: **Summary:** \n **Location:** \n **Start Time:** 2024-11-05T10:00:00-07:00 \n  **End Time:** 2024-11-09T10:00:00-07:00 "
            )
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=self.schedule_meeting_thread.id
                )
                print(messages.data[0].content[0].text.value)
                text = messages.data[0].content[0].text.value
            summary_match = re.search(r"\*\*Summary:\*\*\s*(.*)", text)
            summary = summary_match.group(1).strip() if summary_match else None
            location_match = re.search(r"\*\*Location:\*\*(.*)", text)
            location = location_match.group(1).strip() if location_match else None
            start_time_match = re.search(r"\*\*Start Time:\*\*(.*)", text)
            start_time = start_time_match.group(1).strip() if start_time_match else None
            end_time_match = re.search(r'\*\*End Time:\*\*(.*)', text)
            end_time = end_time_match.group(1).strip() if end_time_match else None
            print ("summary meeting: ", summary, location, start_time, end_time)
            print(f"You have succesffully scheduled meeting with the following information: \n {text}")
            # self.schedule_meeting(summary, location, start_time, end_time)
        elif intent == "search_internet":
            message = client.beta.threads.messages.create(
                thread_id=self.send_email_thread.id,
                role="user",
                content=f"Now the user will search on internet, here's the entered information: '{response}'\n\n .f"
            )
            run = client.beta.threads.runs.create_and_poll(
                thread_id=self.send_email_thread.id,
                assistant_id=self.assistant.id,
                instructions="Please extract and only return the formatted query as the following format: **Query:** \n **Number of Display Results** "
            )
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=self.send_email_thread.id
                )
                print(messages.data[0].content[0].text.value)
                text = messages.data[0].content[0].text.value
            query_match = re.search(r"\*\*Query:\*\*\s*(.*)", text)
            display_match = re.search(r"\*\*Number of Display Result:\*\*\s*(.*)", text)
            query = query_match.group(1).strip() if query_match else None
            display = display_match.group(1).strip() if display_match else None
            print("Query:", query, display)
            return self.search_internet(response)
        elif intent == "read_pdf":
            message = client.beta.threads.messages.create(
                thread_id=self.read_pdf_thread.id,
                role="user",
                content=f"Now the user will read multiple PDF files, and ask questions. Here's the entered information from user: '{response}'\n\n .f"
            )
            run = client.beta.threads.runs.create_and_poll(
                thread_id=self.read_pdf_thread.id,
                assistant_id=self.assistant.id,
                instructions="Please extract and only return the formatted user question as the following format (ignore the part that demands reading pdf, only return the question that based on the pdf) : **Question:** \n "
            )
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=self.read_pdf_thread.id
                )
                print(messages.data[0].content[0].text.value)
                text = messages.data[0].content[0].text.value
            question_match = re.search(r"\*\*Question:\*\*\s*(.*)", text)
            question = question_match.group(1).strip() if question_match else None
            print("Question:", question)
            return self.read_pdf("./", question, "pdfread")
        else:
            return "Your action is beyond my capability, I'm not sure how to help with that."

    def parse_intent(self, user_input):
        try:
            # Query the model to classify the intent
            message = client.beta.threads.messages.create(
              thread_id=self.main_thread.id,
              role="user",
              content=f"Classify the intent of this user request: '{user_input}'\n\n . Just reply in the following four categories: send_email, schedule_meeting, search_internet, read_pdf"
            )
            run = client.beta.threads.runs.create_and_poll(
                thread_id=self.main_thread.id,
                assistant_id=self.assistant.id,
                instructions="Please address the user as Ryan Zhou. The user has a premium account."
            )
            if run.status == 'completed':
                messages = client.beta.threads.messages.list(
                    thread_id=self.main_thread.id
                )
                print(messages.data[0].content[0].text.value)
                return messages.data[0].content[0].text.value
            else:
                print(run.status)
                return run.status
        except Exception as e:
            print(f"Error parsing intent: {e}")
            return "error"

    def send_email(self, recipient, subject, body):
        message = MIMEMultipart()
        message['From'] = self.email_address
        message['To'] = recipient
        message['Subject'] = subject
        message.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP('smtp.gmail.com',
                              587)  # Replace 'smtp.yourserver.com' and 587 with your SMTP server details
        server.ehlo('mylowercasehost')
        server.starttls()
        server.ehlo('mylowercasehost')
        server.login(self.email_address, self.email_password)
        text = message.as_string()
        server.sendmail(self.email_address, recipient, text)
        server.quit()
        print("Email sent to", recipient)

    def schedule_meeting(self, summary, location, start_time, end_time):
        # Extract details and schedule a meeting
        print("Scheduling a meeting...")
        gc = authenticate_google_calendar()

        # create_event('Team Meeting', 'Office 21', '2024-11-05T10:00:00-07:00', '2024-11-05T11:00:00-07:00',
        #              gc, [self.email_address])
        create_event(summary, location, start_time, end_time, gc, [self.email_address])

    def search_internet(self, query):
        # Perform an internet search and return the results
        print("Searching the internet...")
        results = search(query, num_results=10, advanced=True)

        # Print the search results
        for result in results:
            print(f"Title: {result.title}")
            print(f"URL: {result.url}")
            print(f"Description: {result.description}")
            print()


    def read_pdf(self, file_path, prompt, collection_name):
        try:
            # Attempt to get the collection by name
            collection = clientpdf.get_collection(name=collection_name)
            print(f"Collection '{collection_name}' loaded successfully.")
        except Exception as e:
            # If collection doesn't exist, create it
            print(f"Collection '{collection_name}' not found. Creating a new one.")
            collection = clientpdf.create_collection(name=collection_name)

            documents = load_documents(file_path)  # Load documents from a source
            # print("documents:", documents)
            chunks = split_text(documents)  # Split documents into manageable chunks

            # store each document in a vector embedding database
            for i, chunk in enumerate(chunks):
                d = chunk.page_content
                # print(f"Chunk {i}: {d}")
                response = ollama.embeddings(model="llama3.1", prompt=d)
                embedding = response["embedding"]
                # print(f"embedding: {embedding}")
                collection.add(
                    ids=[str(i)],
                    embeddings=[embedding],
                    documents=[d]
                )

        # an example prompt
        # prompt = "What's the main topic of this paper?"
        embedding = ollama.embeddings(
            prompt=prompt,
            model="llama3.1"
        )["embedding"]
        # print(f"embedding: length {len(embedding)})\n{embedding}")

        results = collection.query(
            query_embeddings=[embedding],
            n_results=5
        )

        data1 = results['documents'][0][0]
        data2 = results['documents'][0][1]
        data3 = results['documents'][0][2]
        data4 = results['documents'][0][3]
        data5 = results['documents'][0][4]

        # Combine the data into a single string
        combined_data = f"{data1}\n\n{data2}\n\n{data3}\n\n{data4}\n\n{data5}"

        prompt = f"Using this data: {combined_data}. Respond to this prompt: {prompt}"
        output = ollama.generate(
            model="llama3.1",
            prompt=prompt
        )
        print(f"Reading PDF response:\n{output['response']}")

# Example of using the assistant
print("=================Welcome to personal assistant==============")
assistant = PersonalAIAssistant()
# response = assistant.handle_query("Send an email to John Doe subject 'Meeting Update' with the body 'Please see the attached document for details.'")
# response = assistant.handle_query("Please help me search online about information on the weather of New York City today")
#response = assistant.handle_query("Read the pdfs and answer me what's the main topics of them?")
response = assistant.handle_query("Please help me schedule a metting on Zachary Building with topic of LLM programming, start on 10 am of Nov 13th, 2024 and end on 5 pm of Nov 13th, 2024")

# print(response)
