import os.path
from typing import List, Optional
from datetime import datetime

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from langchain_core.tools import tool

# If modifying these scopes, delete the file token.json.
SCOPES = [
    'https://www.googleapis.com/auth/calendar',
    'https://www.googleapis.com/auth/gmail.modify'
]

def get_credentials():
    """Gets valid user credentials from storage."""
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)
    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                'credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open('token.json', 'w') as token:
            token.write(creds.to_json())
    return creds

def get_calendar_service():
    return build('calendar', 'v3', credentials=get_credentials())

def get_gmail_service():
    return build('gmail', 'v1', credentials=get_credentials())

# --- Calendar Tools ---

@tool
def list_calendars() -> str:
    """Lists the available calendars."""
    service = get_calendar_service()
    calendars_result = service.calendarList().list().execute()
    calendars = calendars_result.get('items', [])
    
    if not calendars:
        return "No calendars found."
    
    result = []
    for cal in calendars:
        result.append(f"ID: {cal['id']}, Summary: {cal['summary']}")
    return "\n".join(result)

@tool
def create_event(summary: str, start_time: str, end_time: str, description: str = "", attendees: List[str] = []) -> str:
    """
    Creates a calendar event.
    Args:
        summary: Title of the event.
        start_time: Start time in ISO format (e.g., '2024-01-01T10:00:00').
        end_time: End time in ISO format.
        description: Optional description.
        attendees: Optional list of email addresses.
    """
    service = get_calendar_service()
    
    event = {
        'summary': summary,
        'description': description,
        'start': {
            'dateTime': start_time,
            'timeZone': 'UTC', # Adjust as needed or parse from input
        },
        'end': {
            'dateTime': end_time,
            'timeZone': 'UTC',
        },
    }
    
    if attendees:
        event['attendees'] = [{'email': email} for email in attendees]

    event = service.events().insert(calendarId='primary', body=event).execute()
    return f"Event created: {event.get('htmlLink')}"

# --- Gmail Tools ---

@tool
def send_gmail_message(to: str, subject: str, body: str) -> str:
    """Sends an email using Gmail."""
    import base64
    from email.mime.text import MIMEText

    service = get_gmail_service()
    
    message = MIMEText(body)
    message['to'] = to
    message['subject'] = subject
    
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    try:
        message = (service.users().messages().send(userId="me", body={'raw': raw_message})
                   .execute())
        return f"Message sent successfully. Id: {message['id']}"
    except Exception as error:
        return f"An error occurred: {error}"

@tool
def draft_gmail_message(to: str, subject: str, body: str) -> str:
    """Creates a draft email in Gmail."""
    import base64
    from email.mime.text import MIMEText

    service = get_gmail_service()
    
    message = MIMEText(body)
    message['to'] = to
    message['subject'] = subject
    
    raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')
    try:
        draft = (service.users().drafts().create(userId="me", body={'message': {'raw': raw_message}})
                 .execute())
        return f"Draft created successfully. Id: {draft['id']}"
    except Exception as error:
        return f"An error occurred: {error}"

# Function to export tools lists
def get_local_tools():
    calendar_tools = [list_calendars, create_event]
    gmail_tools = [send_gmail_message, draft_gmail_message]
    return calendar_tools, gmail_tools