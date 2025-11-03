# Multi-Agent Personal Assistant

This project implements a multi-agent personal assistant using LangGraph. The system is designed to handle tasks such as scheduling calendar events and sending emails by delegating them to specialized agents coordinated by a supervisor agent.

## Features

- **Calendar Agent**: Manages calendar-related tasks via Google Calendar API.
  - Create new calendar events.
  - Check for available time slots for attendees.
- **Email Agent**: Handles email communications via Gmail API.
  - Compose and send emails.
- **Supervisor Agent**: A master agent that interprets user requests and coordinates the other agents to fulfill them. It can handle complex, multi-step tasks that require both scheduling and emailing.
- **Powered by LangGraph**: Leverages the LangGraph framework for building agentic applications with Large Language Models.
- **LLM Integration**: Utilizes commercial GPT or local models for natural language understanding and generation. It is also configured to potentially use Google's GenAI models.
- **Google Workspace Integration**: Uses the [google_workspace_mcp](https://github.com/taylorwilsdon/google_workspace_mcp) MCP server to interact with Google Calendar, Gmail, and other Google Workspace services.

## Requirements

- Python 3.12+
- `uv` package manager
- A commercial LLM key like (OPENAI API, Google API key)
- Google Workspace MCP server running locally

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    This project uses `uv` for dependency management. Install dependencies with:
    ```bash
    uv sync
    ```

3.  **Set up Google Workspace MCP Server:**
    Clone and configure the Google Workspace MCP server:
    ```bash
    git clone https://github.com/taylorwilsdon/google_workspace_mcp
    cd google_workspace_mcp
    # Follow the setup instructions in the google_workspace_mcp repository
    # to configure OAuth credentials and start the MCP server
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root of the project directory and add your API keys:
    ```
    export OPENAI_API_KEY=your-openai-api-key
    export GOOGLE_API_KEY=your-google-api-key
    export LANGSMITH_TRACING=true
    export LANGSMITH_ENDPOINT=https://eu.api.smith.langchain.com
    LANGSMITH_API_KEY="your-langsmith-api-key"  # Optional: for observability
    LANGSMITH_PROJECT="your-project-name"        # Optional: for observability
    GOOGLE_CLOUD_PROJECT="your-google-cloud-project-id"
    ```
    
    **Where to find these credentials:**
    - **OPENAI_API_KEY**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
    - **GOOGLE_API_KEY**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
    - **LANGSMITH_API_KEY** & **LANGSMITH_PROJECT**: Optional - Get from [LangSmith](https://smith.langchain.com/) for monitoring and debugging your agent workflows
    - **GOOGLE_CLOUD_PROJECT**: Your Google Cloud project ID from [Google Cloud Console](https://console.cloud.google.com/) (same project used for Google Workspace APIs)
    
## Usage

This project uses **LangGraph Studio** for interactive development and testing of the multi-agent system.

### Running with LangGraph Studio

To start the development server with LangGraph Studio:

```bash
uv run langgraph dev
```

This will:
- Start the LangGraph development server 
- Open the LangGraph Studio interface in your browser with the link https://smith.langchain.com/studio/?baseUrl=http://127.0.0.1:2024
- Allow you to interact with the agent system through a visual interface
- Provide real-time monitoring of agent decisions and tool calls

### Example Interactions

You can ask the assistant to perform tasks like:
- "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour"
- "Send an email reminder about reviewing the new mockups"
- "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, and send them an email reminder about reviewing the new mockups" (multi-step task)

The supervisor agent will coordinate with the Calendar and Email agents to fulfill your requests.

## Architecture

The system consists of:
- **Supervisor Agent**: Routes tasks to specialized agents based on the request
- **Calendar Agent**: Handles Google Calendar operations via MCP server
- **Email Agent**: Handles Gmail operations via MCP server
- **Google Workspace MCP Server**: Provides secure, standardized access to Google Workspace APIs

## Next Steps
- Add more Google Workspace integrations (Google Drive, Docs, Sheets, etc.)
- Extend the supervisor prompt to cover additional personal assistant skills
- Add support for more complex scheduling scenarios (recurring events, conflict resolution)
- Implement user preferences and context memory
- Add automated tests for agent workflows