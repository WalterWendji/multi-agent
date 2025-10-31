# Multi-Agent Personal Assistant

This project implements a multi-agent personal assistant using LangChain. The system is designed to handle tasks such as scheduling calendar events and sending emails by delegating them to specialized agents coordinated by a supervisor agent.

## Features

- **Calendar Agent**: Manages calendar-related tasks.
  - Create new calendar events.
  - Check for available time slots for attendees.
- **Email Agent**: Handles email communications.
  - Compose and send emails.
- **Supervisor Agent**: A master agent that interprets user requests and coordinates the other agents to fulfill them. It can handle complex, multi-step tasks that require both scheduling and emailing.
- **Powered by LangChain**: Leverages the LangChain framework for building agentic applications with Large Language Models.
- **LLM Integration**: Utilizes OpenAI's GPT models for natural language understanding and generation. It is also configured to potentially use Google's GenAI models.

## Requirements

- Python 3.12+
- An OpenAI API key
- A Google API key

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install dependencies:**
    This project uses the dependencies listed in `pyproject.toml`. You can install them using `pip`:
    ```bash
    pip install .
    ```
    Or if you are using a virtual environment manager like `uv`:
    ```bash
    uv sync
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the root of the project directory and add your API keys:
    ```
    OPENAI_API_KEY="your-openai-api-key"
    GOOGLE_API_KEY="your-google-api-key"
    ```

## Usage

The main logic for the multi-agent system is located in `multi-agent.py`. To run the assistant, you can execute this script directly.

The script contains an example query that demonstrates the supervisor agent's ability to handle a multi-step task. You can uncomment the final block of code in `multi-agent.py` to see it in action.

Example query in the script:
```python
query = ("Schedule a meeting with the design team next Tuesday at 2pm for 1 hour, "
        "and send them an email reminder about reviewing the new mockups."
)
    
for step in supervisor_agent.stream(
    {"messages": [{"role":"user", "content":query}]}
):
    for update in step.values():
        for message in update.get("messages", []):
            message.pretty_print()
```

To run the script:
```bash
python multi-agent.py
```
This will stream the agent's thoughts, actions, and final response to the console.

## Next Steps
- Replace the stub tool implementations with real integrations (Google Calendar, Gmail, etc.).
- Extend the supervisor prompt to cover additional personal assistant skills.
- Add automated tests for tool outputs once real APIs are wired in.