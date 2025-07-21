import os
import base64
from email.mime.text import MIMEText

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

from config import SIMULATOR_EMAIL_ADDRESS, PLATFORM_EMAIL_ADDRESS

# Gmail API SCOPES
SCOPES = [
    'https://www.googleapis.com/auth/gmail.send',
    'https://www.googleapis.com/auth/gmail.readonly',
    'https://www.googleapis.com/auth/gmail.modify'
]

def get_gmail_service():
    """Initializes and returns the Gmail API service."""
    creds = None
    # The file token.json stores the user's access and refresh tokens
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    # If there are no valid credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    try:
        service = build('gmail', 'v1', credentials=creds)
        print("Gmail API service initialized successfully.")
        return service
    except HttpError as error:
        print(f"Error initializing Gmail API service: {error}")
        return None


def create_mime_message(sender, to, subject, message_text):
    """Creates a MIMEText message for sending."""
    message = MIMEText(message_text)
    message['to'] = to
    message['from'] = sender
    message['subject'] = subject
    return {'raw': base64.urlsafe_b64encode(message.as_bytes()).decode()}


def send_email(service, sender_email, receiver_email, subject, body):
    """Sends an email using Gmail API."""
    if not service:
        print("Gmail service not available. Cannot send email.")
        return False

    message = create_mime_message(sender_email, receiver_email, subject, body)
    try:
        sent_message = service.users().messages().send(userId='me', body=message).execute()
        print(f"Email sent successfully to {receiver_email} with subject: '{subject}'")
        return True
    except HttpError as error:
        print(f"Error sending email: {error}")
        return False


def fetch_unread_emails(service, user_id='me', sender_email=PLATFORM_EMAIL_ADDRESS):
    """Fetches unread emails from a specific sender using Gmail API."""
    if not service:
        print("Gmail service not available. Cannot fetch emails.")
        return []

    try:
        # Search for unread emails from the platform
        query = f"is:unread from:{sender_email}"
        response = service.users().messages().list(userId=user_id, q=query).execute()

        messages = []
        if 'messages' in response:
            messages.extend(response['messages'])

        fetched_emails = []
        for msg_id_obj in messages:
            msg_id = msg_id_obj['id']
            msg_data = service.users().messages().get(userId=user_id, id=msg_id, format='full').execute()

            headers = msg_data['payload']['headers']
            subject = next((header['value'] for header in headers if header['name'] == 'Subject'), 'No Subject')
            sender = next((header['value'] for header in headers if header['name'] == 'From'), 'Unknown Sender')

            body = ''
            if 'parts' in msg_data['payload']:
                # Handle multipart messages
                for part in msg_data['payload']['parts']:
                    if part['mimeType'] == 'text/plain' and 'data' in part['body']:
                        body = base64.urlsafe_b64decode(part['body']['data']).decode('utf-8')
                        break
            elif 'body' in msg_data['payload'] and 'data' in msg_data['payload']['body']:
                # Handle single part messages
                body = base64.urlsafe_b64decode(msg_data['payload']['body']['data']).decode('utf-8')

            fetched_emails.append({'id': msg_id, 'sender': sender, 'subject': subject, 'body': body})

            # Mark email as read after processing
            service.users().messages().modify(userId=user_id, id=msg_id, body={'removeLabelIds': ['UNREAD']}).execute()
            print(f"Marked email ID: {msg_id} as read.")

        return fetched_emails
    except HttpError as error:
        print(f"Error fetching emails: {error}")
        return []

