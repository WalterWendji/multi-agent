import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama #For local model
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
import operator


#The AgentState is the graph's state.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set") from None

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set") from None


#model = init_chat_model(model="gpt-5-nano", temperature=0)
#model = init_chat_model("google_genai:gemini-2.5-flash")
model = ChatOpenAI(model="gpt-5-nano", temperature=0)
#model = ChatOllama(model="qwen3:8b", temperature=0)


#Helper function to create an agent node
def create_agent_node(agent, agent_name):
    def agent_node(state):
        result = agent.invoke(state)
        # We convert the agent's response to a ToolMessage
        return {"messages": [ToolMessage(content=result["messages"][-1].text, tool_call_id=agent_name)]}
    return agent_node

def router(state):
    last_message = state["messages"][-1]
    # If the last message is a ToolMessage, it means an agent has just run.
    if isinstance(last_message, ToolMessage):
        return "END"
    if "schedule" in last_message.content.lower() or "meeting" in last_message.content.lower():
        return "calendar"
    if "email" in last_message.content.lower() or "send" in last_message.content.lower():
        return "email"
    else:
        # If no specific agent is needed, we can end.
        return "END"
 

workflow = StateGraph(AgentState)

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
    """Send an email via email API. Requires properly formatted email addresses."""
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
def schedule_event(
    request: str,
    runtime: ToolRuntime
) -> str:
    """Schedule calendar events using natural language."""
    
    original_user_message = next(
        message for message in runtime.state["messages"]
        if message.type == "human"
    )
    prompt = ("You are assisting with the following user inquiry:\n\n"
    f"{original_user_message.text}\n\n"
    "You are tasked with the following sub-request:\n\n"
    f"{request}"
    )
    result = calendar_agent.invoke({
        "messages": [{"role": "user", "content": prompt}]
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

""" query = ("Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups."
)
    
for step in supervisor_agent.stream(
    {"messages": [{"role":"user", "content":query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
             """

calendar_node = create_agent_node(calendar_agent, "calendar") 
email_node = create_agent_node(email_agent, "email")

workflow.add_node("calendar", calendar_node)
workflow.add_node("email", email_node)
supervisor_node = create_agent_node(supervisor_agent, "supervisor")
workflow.add_node("supervisor", supervisor_node)


workflow.set_entry_point("supervisor")
workflow.add_conditional_edges(
    "supervisor",
    router,
    {
        "calendar": "calendar",
        "email": "email",
        "END": END,
    }
)

workflow.add_edge('calendar', END)
workflow.add_edge('email', END)

app = workflow.compile()


def run_multi_agent():
    # Example: User request requiring both calendar and email coordination
    user_request = (
        "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups."
    )

    print("User Request:", user_request)
    print("\n" + "="*80 + "\n")

    """ for step in supervisor_agent.stream(
        {"messages": [{"role": "user", "content": user_request}]}
    ):
        for update in step.values():
            for message in update.get("messages", []):
                message.pretty_print() """

    for event in app.stream({"messages": [HumanMessage(content=user_request)]}):
        for value in event.values():
            print("---")
            if "messages" in value:
                for msg in value["messages"]:
                    if hasattr(msg, 'pretty_print'):
                        msg.pretty_print()
                    else:
                        print(msg)
if __name__ == "__main__":
    run_multi_agent()