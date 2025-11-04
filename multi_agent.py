import os
import operator
import asyncio
import uuid
import sys
from contextlib import asynccontextmanager

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
_app_instance = None
calendar_mcp_tools = None
gmail_mcp_tools = None
_mcp_session = None
_mcp_stdio_context = None




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


# Debug: Track module loading
_module_load_id = str(uuid.uuid4())[:8]
print(f"\n{'='*60}", file=sys.stderr)
print(f"[MODULE LOAD] multi_agent.py loaded - ID: {_module_load_id}", file=sys.stderr)
print(f"{'='*60}\n", file=sys.stderr)

#Helper function to create an agent node
def create_agent_node(agent, agent_name):
    async def agent_node(state):
        result = await agent.ainvoke(state)
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
    
    print(f"[ROUTER] Called - Message count: {len(messages)}", file=sys.stderr)
    
    # Get the original user request 
    original_request = None
    for m in messages:
        if isinstance(m, HumanMessage):
            original_request = m.content.lower() if m.content else ""
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
 



@asynccontextmanager
async def mcp_connection_manager():
    """Async context manager that keeps MCP connection alive.
    This ensures the session stays open throughout the application execution."""
    global _mcp_session, calendar_mcp_tools, gmail_mcp_tools
    
    
    
    server_params = StdioServerParameters(
        command="uv",
        args=["run", "--directory", "../google_workspace_mcp", "main.py", "--single-user"]
    )
    
    
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            _mcp_session = session
            
            # Get all tools from the server
            all_tools = await load_mcp_tools(session)
            
            # Filter calendar tools
            calendar_tool_names = [
                "create_event", "modify_event","list_calendars", 
                "get_event"
            ]
            
            calendar_mcp_tools = [
                tool for tool in all_tools 
                if any(name in tool.name.lower() for name in calendar_tool_names)
            ]
            
            # Filter gmail tools
            gmail_tool_names = [
                "send_gmail_message","draft_gmail_message"
            ]
            gmail_mcp_tools = [
                tool for tool in all_tools 
                if any(name in tool.name.lower() for name in gmail_tool_names)
            ]
            
            print(f"Loaded {len(calendar_mcp_tools)} calendar tools")
            print(f"Loaded {len(gmail_mcp_tools)} gmail tools")
            
            # Verify OAuth authentication is ready before proceeding
            await verify_mcp_authentication(session, all_tools)
            
            # Yield control while keeping connection alive
            try:
                yield session
            finally:
                # Connection will be closed when context exits
                _mcp_session = None
        
        

async def verify_mcp_authentication(session, all_tools, max_retries=5, initial_delay=2):
    """Verify that OAuth authentication is ready by testing a lightweight tool call.
    
    Args:
        session: The MCP client session
        all_tools: List of all available tools
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds before retrying
    
    Raises:
        RuntimeError: If authentication cannot be verified after all retries
    """
    # Find a lightweight tool to test authentication (prefer list_calendars)
    test_tool = None
    for tool in all_tools:
        if "list_calendars" in tool.name.lower():
            test_tool = tool
            break
    
    # If no list_calendars, try any calendar tool or fall back to first tool
    if not test_tool:
        for tool in all_tools:
            if "calendar" in tool.name.lower():
                test_tool = tool
                break
    
    if not test_tool and all_tools:
        test_tool = all_tools[0]
    
    if not test_tool:
        print("[AUTH] Warning: No tools available to verify authentication", file=sys.stderr)
        return
    
    delay = initial_delay
    
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[AUTH] Verifying authentication (attempt {attempt}/{max_retries})...", file=sys.stderr)
            # Try calling the test tool with empty/minimal arguments
            # Most list tools don't require arguments
            result = await session.call_tool(test_tool.name, {})
            print(f"[AUTH] Authentication verified successfully", file=sys.stderr)
            return
        except Exception as e:
            error_msg = str(e).lower()
            # Check if error is authentication-related
            if "oauth" in error_msg or "authenticated" in error_msg or "authentication" in error_msg:
                if attempt < max_retries:
                    print(f"[AUTH] Authentication not ready yet, waiting {delay}s before retry...", file=sys.stderr)
                    await asyncio.sleep(delay)
                    delay = min(delay * 1.5, 10)  # Exponential backoff, max 10s
                else:
                    raise RuntimeError(
                        f"OAuth authentication verification failed after {max_retries} attempts. "
                        f"Please ensure the Google Workspace MCP server has completed OAuth authentication. "
                        f"The MCP server may need to complete the OAuth flow in single-user mode. "
                        f"Last error: {str(e)}"
                    ) from e
            else:
                # If it's not an auth error, authentication might be OK but tool call failed for other reasons
                print(f"[AUTH] Tool call returned non-auth error, assuming authentication is ready", file=sys.stderr)
                return

async def initialize_mcp_tools():
    """Initialize MCP tools from remote servers.
    This function must be called within the mcp_connection_manager context."""
    # Tools are initialized within the context manager
    # This function is kept for compatibility but does nothing
    pass
    


def initialize_mcp_tools_sync():
    """This function is deprecated - initialization happens in mcp_connection_manager."""
    # This is a no-op now since initialization happens in the context manager
    pass




CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm')"
    "into proper ISO datetime formats. "
    "Use get_available_time_slots to check availability when needed. "
    "Use create_calendar_event to schedule events. "
    "Always confirm what was scheduled in your final response. "
)

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "Use send_email to send the message. "
    "Always confirm what was sent in your final response. "
)

SUPERVISOR_PROMPT  = (
    "You are a helpful personal assistant coordinator. "
    "Analyze user requests and determine which specialized agent should handle them. "
    "Respond with clear routing hints: "
    "- If the request involves scheduling, meetings, or calendar events, mention 'calendar' or 'schedule'"
    "- If the request involves sending emails, messages, or notifications, mention 'email' or 'send'"
    "For requests involving multiple actions, prioritize the first action or the most important one. "
    "Keep your response concise and focused on routing."
)

def create_graph():
    """Create and return the compiled LangGraph application."""
    global calendar_mcp_tools, gmail_mcp_tools, _app_instance
    
    if _app_instance is not None:
        return _app_instance
    
    # Tools must be initialized before calling create_graph()
    # This should happen within the mcp_connection_manager context
    if calendar_mcp_tools is None or gmail_mcp_tools is None:
        raise RuntimeError("MCP tools not initialized. Call create_graph() within mcp_connection_manager context.")
    
    calendar_agent = create_agent(
        model,
        tools=calendar_mcp_tools,
        system_prompt=CALENDAR_AGENT_PROMPT,
    )

    email_agent = create_agent(
        model,
        tools=gmail_mcp_tools,
        system_prompt=EMAIL_AGENT_PROMPT,
    )

    print(f"[CREATE_GRAPH] Called in module {_module_load_id}", file=sys.stderr)
    print(f"[CREATE_GRAPH] _app_instance is None: {_app_instance is None}", file=sys.stderr)
    #Create nodes
    calendar_node = create_agent_node(calendar_agent, "calendar") 
    email_node = create_agent_node(email_agent, "email")

    async def supervisor_node(state):
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
        response = await model.ainvoke(messages_with_system)
        return {"messages": [response]}



    workflow = StateGraph(AgentState)

    #Build workflow
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
            "email": "email",
            "END": END,
        }
    )
    workflow.add_conditional_edges(
        'email',
        router,
        {
            "calendar": "calendar",
            "END": END,
        }
    )

    _app_instance = workflow.compile()
    return _app_instance


#app = create_graph()
app = None


async def run_multi_agent():
    global app
    
    # Example: User request requiring both calendar and email coordination
    user_request = (
        "send an email to d.wendjwalter@gmail.com with the subject 'Meeting tomorrow'. He should come tomorrow at 10am in the office. My google account email address is satelliteplus12@gmail.com"
    )

    print("User Request:", user_request)
    print("\n" + "="*80 + "\n")

    # Use the context manager to keep MCP connection alive
    async with mcp_connection_manager():
        # Create graph now that tools are initialized
        if app is None:
            app = create_graph()
        
        async for event in app.astream({"messages": [HumanMessage(content=user_request)], "completed_actions": set()}):
            for value in event.values():
                print("---")
                if "messages" in value:
                    for msg in value["messages"]:
                        if hasattr(msg, 'pretty_print'):
                            msg.pretty_print()
                        else:
                            print(msg)
if __name__ == "__main__":
    asyncio.run(run_multi_agent())