import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langchain.agents import create_agent


load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set") from None

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set") from None


#model = init_chat_model(model="gpt-5-nano", temperature=0)
#model = init_chat_model("google_genai:gemini-2.5-flash")
model = init_chat_model("openai:gpt-5-nano")

@tool
def create_calendar_event(
    title: str,
    start_time: str,
    end_time: str,
    attendees: list[str],
    location: str =""
) -> str:
    """Create a calendar event. Requires exact ISO datetime format."""
    #Stub: In practice, this would call Google Calendar API, Outlook API, etc:
    return f"Event created: {title} from {start_time} to {end_time} with {len(attendees)} attendees"

@tool
def send_email(
    to: list[str],
    subject: str,
    body: str,
    cc: list[str] = []
) -> str:
    """Send an email via email API. Requires properly formatted email addressess."""
    # Stub: In practice, this would call SendGrid, Gmail API, etc.
    return f"Email sent to to {',' .join(to)} - Subject: {subject}"

@tool
def get_available_time_slots(
    attendees: list[str],
    date: str,
    duration_minutes: int
) -> list[str]:
    """Check calendar availability for given attendees on a specific date."""
    #Stub: In practice, this would query calendar APIs
    return ["09:00", "14:00", "16:00"]


CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm')"
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response. "
)

calendar_agent = create_agent(
    model,
    tools=[create_calendar_event, get_available_time_slots],
    system_prompt=CALENDAR_AGENT_PROMPT,
)

""" query= "Schedule a team meeting next Tuesday at 2pm for 1 hour"

for step in calendar_agent.stream(
    {"messages": [{"role": "user", "content":query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print() """
            

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response. "
)

email_agent = create_agent(
    model,
    tools=[send_email],
    system_prompt=EMAIL_AGENT_PROMPT,
)

""" query = "Send the design team a reminder about reviewing the new mockups"

for step in email_agent.stream(
    {"messages": [{"role": "user", "content":query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
             """

@tool
def schedule_event(request: str) -> str:
    """Schedule calendar events using natural language.

    Use this when the user wants to create, modify, or check calendar appointments.
    Handles date/time parsing, availability checking, and event creation.

    Input: Natural language scheduling request (e.g., 'meeting with design team
    next Tuesday at 2pm')
    """
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text


@tool
def manage_email(request: str) -> str:
    """Send emails using natural language.

    Use this when the user wants to send notifications, reminders, or any email
    communication. Handles recipient extraction, subject generation, and email
    composition.

    Input: Natural language email request (e.g., 'send them a reminder about
    the meeting')
    """
    result = email_agent.invoke({
        "messages": [{"role": "user", "content": request}]
    })
    return result["messages"][-1].text

SUPERVISOR_PROMPT  = (
    "You are a helpful personal assistant. "
    "You can schedule calendar events and send emails. "
    "Break down user requests into appropriate tool calls and coordinate the results. "
    "When a request involves multiple actions, use multiple tools in sequence."
)

supervisor_agent = create_agent(
    model,
    tools=[schedule_event, manage_email],
    system_prompt=SUPERVISOR_PROMPT,
)

query = ("Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups."
)
    
for step in supervisor_agent.stream(
    {"messages": [{"role":"user", "content":query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()