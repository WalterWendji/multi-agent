import os
import operator
import asyncio

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_ollama import ChatOllama #For local model
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain.tools import tool, ToolRuntime
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import tools_condition
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langgraph.checkpoint.memory import MemorySaver


#Store MCP sessions
calendar_mcp_tools = None
gmail_mcp_tools = None




#The AgentState is the graph's state.
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    completed_actions: set[str]  # Track completed agent actions




load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set") from None

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set") from None


#model = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
model = ChatOpenAI(model="gpt-5-nano", temperature=0)
#model = ChatOllama(model="qwen3:8b", temperature=0)


#Helper function to create an agent node
def create_agent_node(agent, agent_name):
    def agent_node(state):
        result = agent.invoke(state)
        # Get messages from agent result
        messages = result["messages"]
        # Get current completed actions from state, or initialize empty set
        completed_actions = state.get("completed_actions", set()).copy()
        # Mark this agent as completed (only for calendar and email agents)
        if agent_name in ['calendar', 'email']:
            completed_actions.add(agent_name)
        return {"messages": messages, "completed_actions": completed_actions}
    return agent_node

def router(state):
    """
    Routes based on the supervisor agent's response or completion status.
    Checks for tool calls first, then falls back to content analysis.
    Handles multi-action requests by tracking completed actions.
    """
    messages = state["messages"]
    
    # Get completed actions from state
    completed_actions = state.get("completed_actions", set())
    
    # Get the original user request
    original_request = None
    for m in messages:
        if isinstance(m, HumanMessage):
            original_request = m.content if m.content else ""
            break
            
    # SHORT-CIRCUIT: If no actionable keywords in original request AND no completed actions,
    # return END immediately (don't route to calendar/email)
    if not completed_actions and original_request:
        has_actionable = any(kw in original_request for kw in [
            "schedule", "meeting", "calendar", "appointment", "book", "reserve",
            "email", "send", "message", "notify", "reminder", "mail"
        ])
        if not has_actionable:
            return "END"
    
    # Check if we just completed an action and need to route to the next one
    # Look at the last few messages to see if a calendar/email agent just ran
    if completed_actions:
        # Check if there are more actions pending based on original request
        if original_request:
            has_calendar = any(kw in original_request for kw in ["schedule", "meeting", "calendar", "appointment"])
            has_email = any(kw in original_request for kw in ["email", "send", "message", "notify", "reminder"])
            
            # If we just completed calendar and email is still pending, route to email
            if 'calendar' in completed_actions and has_email and 'email' not in completed_actions:
                return "email"
            # If we just completed email and calendar is still pending, route to calendar
            elif 'email' in completed_actions and has_calendar and 'calendar' not in completed_actions:
                return "calendar"
            # If both are completed, end
            elif ('calendar' in completed_actions and has_calendar) and ('email' in completed_actions and has_email):
                return "END"
    
    # Look for the most recent AI message from supervisor
    last_ai_message = None
    content = ""
    
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            last_ai_message = msg
            break
    if last_ai_message:
        content = (last_ai_message.content or "").lower()

        # Check for tool calls in the AI message
        if hasattr(last_ai_message, 'tool_calls') and last_ai_message.tool_calls:
            tool_calls = last_ai_message.tool_calls
            for tool_call in tool_calls:
                tool_name = tool_call.get('name', '') if isinstance(tool_call, dict) else getattr(tool_call, 'name', '')
                if tool_name == 'schedule_event':
                    return "calendar"
                elif tool_name == 'manage_email':
                    return "email"
        
    
    
    # Check for calendar-related keywords
    calendar_keywords = ["schedule", "meeting", "calendar", "appointment", "book", "reserve"]
    has_calendar = any(keyword in content for keyword in calendar_keywords)
    
    # Check for email-related keywords
    email_keywords = ["email", "send", "message", "notify", "reminder", "mail"]
    has_email = any(keyword in content for keyword in email_keywords)
    
    # Prioritize calendar if both are present and not completed
    if has_calendar and 'calendar' not in completed_actions:
        return "calendar"
    elif has_email and 'email' not in completed_actions:
        return "email"
    
    # If no clear routing signal from supervisor, check original request as fallback
    if original_request:
        has_calendar = any(kw in original_request for kw in ["schedule", "meeting", "calendar", "appointment"])
        has_email = any(kw in original_request for kw in ["email", "send", "message", "notify", "reminder"])
        
        if has_calendar and 'calendar' not in completed_actions:
            return "calendar"
        if has_email and 'email' not in completed_actions:
            return "email"
    
    # If no clear routing signal, end the workflow
    return "END"
 



async def setup_workspace_mcp_tools():
    """Connect to local Google Workspace MCP server and get all tools."""
    
    
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", "../google_workspace_mcp", "main.py", "--single-user"]
    )
    
    
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            
            # Get all tools from the server
            tools = await load_mcp_tools(session)
    
            
            return tools, session
        
        

async def initialize_mcp_tools():
    """Initialize MCP tools from remote servers."""
    global calendar_mcp_tools, gmail_mcp_tools
    
    #Run async functions in sync context
    # Connect to the workspace MCP server
    all_tools, session = await setup_workspace_mcp_tools()
    
    calendar_tool_names = [
        "create_event", "modify_event","list_calendars", 
        "get_event"
    ]
    
    calendar_mcp_tools = [
        tool for tool in all_tools 
        if any(name in tool.name.lower() for name in calendar_tool_names)
    ]
    
    # Filter tools for gmail (based on tool names - adjust as needed)
    gmail_tool_names = [
        "send_gmail_message","draft_gmail_message"
    ]
    gmail_mcp_tools = [
        tool for tool in all_tools 
        if any(name in tool.name.lower() for name in gmail_tool_names)
    ]
    
    print(f"Loaded {len(calendar_mcp_tools)} calendar tools")
    print(f"Loaded {len(gmail_mcp_tools)} gmail tools")
    


def initialize_mcp_tools_sync():
    """Wrapper to run async initialization synchronously."""
    asyncio.run(initialize_mcp_tools())

# Initialize at module level (or move to a function called before creating agents)
initialize_mcp_tools_sync()




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
    tools=calendar_mcp_tools,
    system_prompt=CALENDAR_AGENT_PROMPT,
)

            

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response. "
)

email_agent = create_agent(
    model,
    tools=gmail_mcp_tools,
    system_prompt=EMAIL_AGENT_PROMPT,
)


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
    "You are a helpful personal assistant coordinator. "
    "Analyze user requests and determine which specialized agent should handle them. "
    "Respond with clear routing hints: "
    "- If the request involves scheduling, meetings, or calendar events, mention 'calendar' or 'schedule'"
    "- If the request involves sending emails, messages, or notifications, mention 'email' or 'send'"
    "For requests involving multiple actions, prioritize the first action or the most important one. "
    "Keep your response concise and focused on routing."
)



calendar_node = create_agent_node(calendar_agent, "calendar") 
email_node = create_agent_node(email_agent, "email")

def supervisor_node(state):
    """Supervisor node that uses the model with system prompt."""
    messages = state["messages"]
    
    
    #Add system prompt if not already present
    has_system = any(isinstance(m, SystemMessage) for m in messages)
    if not has_system:
        system_msg = SystemMessage(content=SUPERVISOR_PROMPT)
        messages_with_system = [system_msg] + list(messages)
    else:
        messages_with_system = messages
    
    #Call model directly
    response = model.invoke(messages_with_system)
    return {"messages": [response]}



workflow = StateGraph(AgentState)


workflow.add_node("calendar", calendar_node)
workflow.add_node("email", email_node)
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

# Route calendar and email nodes back to supervisor to check for additional actions
workflow.add_conditional_edges(
    'calendar',
    router,
    {
        "calendar": "calendar",
        "email": "email",
        "END": END,
    }
)
workflow.add_conditional_edges(
    'email',
    router,
    {
        "calendar": "calendar",
        "email": "email",
        "END": END,
    }
)

memory = MemorySaver()
app = workflow.compile()

