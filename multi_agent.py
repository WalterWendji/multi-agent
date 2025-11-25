import os
import asyncio
import uuid
import sys
from typing import TypedDict, Annotated, Sequence

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_mcp_adapters.tools import load_mcp_tools
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# Store MCP sessions
_app_instance = None
_mcp_session = None
_tools_cache = None

class GraphState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    completed_actions: set[str]  # Track completed agent actions

load_dotenv()
if not os.environ.get("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY environment variable not set") from None

if not os.environ.get("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY environment variable not set") from None

# Get user's Google email from environment variable
user_google_email = os.environ.get("USER_GOOGLE_EMAIL")
if not user_google_email:
    raise ValueError("USER_GOOGLE_EMAIL environment variable not set. Please set it in your .env file.") from None


#model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
model = ChatOpenAI(model="gpt-5.1", temperature=0, reasoning_effort="low")
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


#Routes based on the supervisor agent's response and completion status.
#Primary routing is based on the supervisor's analysis, with fallback to keyword matching.
def router(state):
    """
    Routes based on the supervisor agent's response or completion status.
    First checks the supervisor's response for routing intent, then falls back to keyword matching.
    Handles multi-action requests by tracking completed actions.
    """
    messages = state["messages"]
    
    # Get completed actions from state
    completed_actions = state.get("completed_actions", set())
    
    print(f"[ROUTER] Called - Message count: {len(messages)}, Completed actions: {completed_actions}", file=sys.stderr)
    
    # Get the original user request (most recent human message)
    original_request = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            original_request = m.content if m.content else ""
            break
    
    if not original_request:
        print("[ROUTER] No original request found, ending", file=sys.stderr)
        return "END"
    
    original_request_lower = original_request
    print(f"[ROUTER] Original request: {original_request[:100]}...", file=sys.stderr)
    
    # Get the supervisor's response (most recent AIMessage, which should be from supervisor)
    supervisor_response = None
    supervisor_response_lower = None
    for m in reversed(messages):
        if isinstance(m, AIMessage) and not isinstance(m, ToolMessage):
            supervisor_response = m.content if m.content else ""
            supervisor_response_lower = supervisor_response.lower()
            print(f"[ROUTER] Supervisor response: {supervisor_response[:200]}...", file=sys.stderr)
            break
    
    # Define keywords for routing
    calendar_keywords = ["schedule", "meeting", "calendar", "appointment", "book", "reserve", "event", "reminder", "scheduling"]
    email_keywords = ["email", "send", "message", "notify", "reminder", "mail", "e-mail", "compose", "mailing"]
    
    # First, try to determine routing from supervisor's response
    has_calendar = False
    has_email = False
    
    if supervisor_response_lower:
        # Check supervisor response for routing indicators
        supervisor_mentions_calendar = any(
            kw in supervisor_response_lower for kw in 
            ["calendar", "schedule", "scheduling", "meeting", "appointment", "calendar_agent"]
        )
        supervisor_mentions_email = any(
            kw in supervisor_response_lower for kw in 
            ["email", "mail", "email_agent", "send", "compose"]
        )
        supervisor_mentions_general = any(
            kw in supervisor_response_lower for kw in 
            ["general", "question", "answer", "help", "information"]
        )
        
        # If supervisor clearly indicates routing, use that
        if supervisor_mentions_calendar and not supervisor_mentions_general:
            has_calendar = True
            print("[ROUTER] Supervisor indicated calendar routing", file=sys.stderr)
        if supervisor_mentions_email and not supervisor_mentions_general:
            has_email = True
            print("[ROUTER] Supervisor indicated email routing", file=sys.stderr)
        if supervisor_mentions_general and not (supervisor_mentions_calendar or supervisor_mentions_email):
            print("[ROUTER] Supervisor indicated general question, ending", file=sys.stderr)
            return "END"
    
    # Fallback to keyword matching on original request if supervisor didn't clearly indicate routing
    if not has_calendar and not has_email:
        has_calendar = any(kw in original_request_lower for kw in calendar_keywords)
        has_email = any(kw in original_request_lower for kw in email_keywords)
        print(f"[ROUTER] Fallback to keyword matching - Calendar: {has_calendar}, Email: {has_email}", file=sys.stderr)
    
    # If no actionable keywords, return END
    if not has_calendar and not has_email:
        print("[ROUTER] No actionable keywords found, ending", file=sys.stderr)
        return "END"
    
    # Handle multi-action routing: check if we need to route to the next action
    if completed_actions:
        # If we just completed calendar and email is still pending, route to email
        if 'calendar' in completed_actions and has_email and 'email' not in completed_actions:
            print("[ROUTER] Routing to email (calendar completed)", file=sys.stderr)
            return "email"
        # If we just completed email and calendar is still pending, route to calendar
        elif 'email' in completed_actions and has_calendar and 'calendar' not in completed_actions:
            print("[ROUTER] Routing to calendar (email completed)", file=sys.stderr)
            return "calendar"
        # If both are completed, end
        elif (('calendar' in completed_actions and has_calendar) and 
              ('email' in completed_actions and has_email)):
            print("[ROUTER] All actions completed, ending", file=sys.stderr)
            return "END"
        # If one is completed but the other wasn't requested, end
        elif ('calendar' in completed_actions and not has_email) or ('email' in completed_actions and not has_calendar):
            print("[ROUTER] Requested action completed, ending", file=sys.stderr)
            return "END"
    
    # Primary routing logic: route based on supervisor analysis or keyword matching
    # Prioritize calendar if both are present (calendar usually comes first in workflow)
    if has_calendar and 'calendar' not in completed_actions:
        print("[ROUTER] Routing to calendar", file=sys.stderr)
        return "calendar"
    elif has_email and 'email' not in completed_actions:
        print("[ROUTER] Routing to email", file=sys.stderr)
        return "email"
    
    # Fallback: if we get here and have actions but they're completed, end
    print("[ROUTER] No routing match found, ending", file=sys.stderr)
    return "END"
 


def _get_mcp_server_params():
    """Get MCP server parameters. Centralized for consistency."""
    return StdioServerParameters(
        command="uv",
        args=["run", "--directory", "../google_workspace_mcp", "main.py", "--single-user"]
    )


async def _initialize_mcp_tools_from_session(session):
    """Initialize and filter MCP tools from a session.
    
    Args:
        session: The MCP client session
        
    Returns:
        tuple: (calendar_tools, gmail_tools, all_tools)
    """
    # Get all tools from the server
    all_tools = await load_mcp_tools(session)
    
    # Filter calendar tools
    calendar_tool_names = [
        "create_event", "modify_event", "list_calendars", "get_event"
    ]
    
    calendar_tools = [
        tool for tool in all_tools 
        if any(name in tool.name.lower() for name in calendar_tool_names)
    ]
    
    # Filter gmail tools
    gmail_tool_names = [
        "send_gmail_message", "draft_gmail_message"
    ]
    gmail_tools = [
        tool for tool in all_tools 
        if any(name in tool.name.lower() for name in gmail_tool_names)
    ]
    
    print(f"[MCP] Loaded {len(calendar_tools)} calendar tools", file=sys.stderr)
    print(f"[MCP] Loaded {len(gmail_tools)} gmail tools", file=sys.stderr)
    
    # Log tool names for debugging
    if calendar_tools:
        print(f"[MCP] Calendar tool names: {[t.name for t in calendar_tools]}", file=sys.stderr)
    if gmail_tools:
        print(f"[MCP] Gmail tool names: {[t.name for t in gmail_tools]}", file=sys.stderr)
    
    return calendar_tools, gmail_tools, all_tools


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
                        "Please ensure the Google Workspace MCP server has completed OAuth authentication. "
                        "The MCP server may need to complete the OAuth flow in single-user mode. "
                        f"Last error: {str(e)}"
                    ) from e
            else:
                # If it's not an auth error, authentication might be OK but tool call failed for other reasons
                print(f"[AUTH] Tool call returned non-auth error, assuming authentication is ready", file=sys.stderr)
                return

async def get_mcp_tools():
    """Lazy initialization on the SAME loop as the app"""
    global _mcp_session, _tools_cache
    
    if _tools_cache:
        return _tools_cache

    server_params = _get_mcp_server_params()
    
    # Connect directly on the current loop
    client = stdio_client(server_params)
    read, write = await client.__aenter__()
    session = ClientSession(read, write)
    await session.__aenter__()
    await session.initialize()
    
    _mcp_session = session # Store to keep alive
    
    # Load tools
    calendar_tools, gmail_tools, all_tools = await _initialize_mcp_tools_from_session(session)
    
    # Verify authentication
    await verify_mcp_authentication(session, all_tools)
    
    _tools_cache = (calendar_tools, gmail_tools)
    return _tools_cache

CALENDAR_AGENT_PROMPT = (
    "You are a calendar scheduling assistant. "
    "Parse natural language scheduling requests (e.g., 'next Tuesday at 2pm') into proper ISO datetime ormats. "
    "When the user asks to schedule a meeting or create an event, you MUST use the create_event tool to actually create it. "
    "You may optionally use the get_event or list_calendars tools to check availability irst, but you MUST call create_event to actually schedule the meeting. "
    "Do not just acknowledge the request - you must actually execute the create_event tool with all required parameters (title, start time, end time, etc.). "
    "The user's Google email is " + str(user_google_email) + ". "
    "Always confirm what was scheduled in your final response with details about the event title, time, and participants if any. "
)

EMAIL_AGENT_PROMPT = (
    "You are an email assistant. "
    "Compose professional emails based on natural language requests. "
    "Extract recipient information and craft appropriate subject lines and body text. "
    "When the user wants to send an email, you MUST use the send_gmail_message tool to actually send it. "
    "Only use draft_gmail_message if the user explicitly asks to draft/save a draft without sending. "
    "If the user asks to send an email, send it immediately using send_gmail_message - do not just draft it. "
    "The user's Google email is " + str(user_google_email) + ". "
    "Always confirm what was sent in your final response with details about the recipient and subject. "
)

SUPERVISOR_PROMPT  = (
    "You are a helpful personal assistant coordinator. "
    "Analyze user requests and determine which specialized agent should handle them. "
    "For scheduling/calendar requests, explicitly mention 'calendar' or 'scheduling' in your response to route to the calendar_agent. "
    "For email requests, explicitly mention 'email' or 'mail' in your response to route to the email_agent. "
    "For general questions that don't require calendar or email actions, mention 'general question' or 'information' in your response. "
    "Keep your responses brief and friendly. "
    "Always clearly indicate which agent should handle the request by mentioning the relevant keywords (calendar, email, or general)."
)

def create_graph():
    """Create and return the compiled LangGraph application."""
    global _app_instance
    
    if _app_instance is not None:
        print(f"[CREATE_GRAPH] Returning cached instance", file=sys.stderr)
        return _app_instance
    
    # Initialize tools on the main loop
    # This blocks the main thread briefly to set up connection if not already done
    # Use nest_asyncio to allow nested event loops if we are already in one (e.g. Jupyter or Uvicorn)
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        if loop.is_running():
            # If loop is running, we might be in a notebook or server.
            # Ideally we should await, but create_graph is sync.
            # nest_asyncio + run_until_complete handles this.
            cal_tools, gmail_tools = loop.run_until_complete(get_mcp_tools())
        else:
            cal_tools, gmail_tools = loop.run_until_complete(get_mcp_tools())
    except Exception as e:
        print(f"[CREATE_GRAPH] Error initializing tools: {e}", file=sys.stderr)
        raise

    
    # Log available tools for debugging
    print(f"[CREATE_GRAPH] Calendar tools: {[t.name for t in cal_tools]}", file=sys.stderr)
    print(f"[CREATE_GRAPH] Gmail tools: {[t.name for t in gmail_tools]}", file=sys.stderr)
    
    calendar_agent = create_agent(
        model,
        tools=cal_tools,
        system_prompt=CALENDAR_AGENT_PROMPT,
    )

    email_agent = create_agent(
        model,
        tools=gmail_tools,
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



    graph = StateGraph(GraphState)

    #Build graph
    graph.add_node("calendar", calendar_node)
    graph.add_node("email", email_node)
    graph.add_node("supervisor", supervisor_node)


    #graph.add_edge(START, "supervisor")
    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        router,
        {
            "calendar": "calendar",
            "email": "email",
            "END": END,
        }
    )

    # Route calendar and email nodes back to supervisor to check for additional actions
    graph.add_conditional_edges(
        'calendar',
        router,
        {
            "email": "email",
            "END": END,
        }
    )
    graph.add_conditional_edges(
        'email',
        router,
        {
            "calendar": "calendar",
            "END": END,
        }
    )

    # Streaming should work automatically with Studio's built-in persistence
    _app_instance = graph.compile()
    print(f"[CREATE_GRAPH] Graph compiled successfully", file=sys.stderr)
    return _app_instance

# Create graph after MCP tools are initialized - this is for LangGraph Studio
try:
    graph = create_graph()
    print("[MODULE] Graph created and exported as 'graph'", file=sys.stderr)
except Exception as e:
    print(f"[MODULE] Error creating graph: {e}", file=sys.stderr)
    print("[MODULE] Graph will be created on first use", file=sys.stderr)
    graph = None
