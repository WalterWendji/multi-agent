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
- **Google Workspace Integration**: Directly integrates with Google Calendar and Gmail APIs for seamless operations.

## Requirements

- Python 3.12+
- `uv` package manager
- A commercial LLM key (OpenAI API key)
- Google Cloud Project with Calendar and Gmail APIs enabled

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    This project uses `uv` for dependency management.
    ```bash
    uv sync
    ```
    This will install all required packages.

3.  **Set up Google Cloud Credentials:**
    - Go to the [Google Cloud Console](https://console.cloud.google.com/).
    - Create a project and enable the **Google Calendar API** and **Gmail API**.
    - Go to **APIs & Services > Credentials**.
    - Create **OAuth 2.0 Client IDs** (Application type: Desktop app).
    - Download the JSON file, rename it to `credentials.json`, and place it in the root directory of this project.

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
    export USER_GOOGLE_EMAIL="your-gmail-address-the-ai-agent-will-have-access-to"
    ```
    
    **Where to find these credentials:**
    - **OPENAI_API_KEY**: Get from [OpenAI Platform](https://platform.openai.com/api-keys)
    - **GOOGLE_API_KEY**: Get from [Google AI Studio](https://aistudio.google.com/app/apikey)
    - **LANGSMITH_API_KEY** & **LANGSMITH_PROJECT**: Optional - Get from [LangSmith](https://smith.langchain.com/) for monitoring and debugging your agent workflows
    - **GOOGLE_CLOUD_PROJECT**: Your Google Cloud project ID from [Google Cloud Console](https://console.cloud.google.com/)

## Usage

This project uses **LangGraph Studio** for interactive development and testing of the multi-agent system.

### Running with LangGraph Studio

To start the development server with LangGraph Studio:

```bash
uv run langgraph dev
```

This will:
- Start the LangGraph development server 
- Open the LangGraph Studio interface in your browser
- Allow you to interact with the agent system through a visual interface
- Provide real-time monitoring of agent decisions and tool calls

**Note:** On the first run, a browser window will open asking you to authorize the application to access your Google Calendar and Gmail. Follow the prompts to allow access. A `token.json` file will be created to store your session.

### Example Interactions

You can ask the assistant to perform tasks like:
- "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour"
- "Send an email reminder about reviewing the new mockups"
- "Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, and send them an email reminder about reviewing the new mockups" (multi-step task)

The supervisor agent will coordinate with the Calendar and Email agents to fulfill your requests.

## Architecture

The system consists of:
- **Supervisor Agent**: Routes tasks to specialized agents based on the request
- **Calendar Agent**: Handles Google Calendar operations via direct API calls
- **Email Agent**: Handles Gmail operations via direct API calls
- **Google Tools**: A local module (`google_tools.py`) that handles OAuth authentication and API interactions

## Next Steps
- Add more Google Workspace integrations (Google Drive, Docs, Sheets, etc.)
- Extend the supervisor prompt to cover additional personal assistant skills
- Add support for more complex scheduling scenarios (recurring events, conflict resolution)
- Implement user preferences and context memory
- Add automated tests for agent workflows
