# Openterface-Ops-GUI

## Overview

Openterface-Ops-GUI is a web-based interface for interacting with AI models, specifically designed for UI inspection and automated interaction. It provides a seamless way to communicate with both language models and UI inspection models, enabling users to analyze and interact with UI elements through natural language commands. The application features a ReAct (Reasoning + Acting) agent mode for autonomous task execution with approval mechanisms for dangerous operations.

## Architecture

The application follows a modular architecture with clear separation of concerns:

```
Openterface-Ops-GUI/
├── ops_api.py              # Application entry point
├── config.py               # Global configuration management
├── index.html              # Frontend web interface
├── ui_model_server.py      # Standalone UI-Model server
├── ops_api/                # Backend API module
│   ├── app.py              # FastAPI application factory
│   ├── endpoints.py        # API endpoints implementation
│   ├── models.py           # Pydantic model definitions
│   ├── session.py          # Session management
│   ├── task_manager.py     # ReAct task manager
│   ├── react_context.py    # ReAct context builder
│   └── react_memory.py     # ReAct memory system
├── ops_core/               # Core functionality modules
│   ├── api/                # API client modules
│   │   ├── client.py       # LLM API client
│   │   └── connection.py   # API connection testing
│   ├── i18n/               # Internationalization
│   │   └── translator.py   # Language translator
│   ├── image/              # Image processing
│   │   ├── encoder.py      # Image encoding utilities
│   │   └── drawer.py       # Image drawing utilities
│   ├── image_server/       # Image server client
│   │   └── client.py       # TCP image server client
│   ├── rag/                # RAG functionality
│   │   ├── index_builder.py
│   │   ├── index_loader.py
│   │   ├── readers.py      # Document readers
│   │   └── retriever.py    # Document retriever
│   └── ui_operations/      # UI operations
│       ├── executor.py     # Command executor
│       ├── parser.py       # Response parser
│       ├── ui_ins_client.py # UI-Ins client
│       └── checkbox_detector.py # Checkbox detection
├── i18n/                   # Translation files
│   ├── en.json             # English translations
│   └── zh.json             # Chinese translations
└── tools/                  # Utility tools
```

## Features

- **Chat Interface**: Interactive chat with AI models
- **Image Capture**: Get and display the latest screen image from TCP server
- **UI Element Detection**: Identify and interact with UI elements using UI-Model
- **ReAct Agent Mode**: Autonomous task execution with reasoning and acting cycles
- **Approval System**: Manual, auto, or strict approval policies for dangerous operations
- **Multi-turn Conversations**: Maintain conversation context across multiple messages
- **Document Retrieval (RAG)**: Build and query document indexes
- **Language Support**: Switch between English and Chinese
- **Session Management**: Create and manage multiple sessions with different configurations
- **Checkbox Detection**: Refined clicking on checkbox elements
- **SSE Streaming**: Real-time progress updates via Server-Sent Events

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone <repository-url>
cd Openterface-Ops-GUI
```

2. Install required dependencies:

```bash
pip install -r requirements.txt
```

3. Set environment variables for API keys (depending on your model setup):

**Windows:**
```cmd
set LLM_API_KEY=your_llm_api_key
set UI_API_KEY=your_ui_api_key
```

**Linux/Mac:**
```bash
export LLM_API_KEY=your_llm_api_key
export UI_API_KEY=your_ui_api_key
```

## Configuration

### Model Configuration

The application supports two types of models that can be configured through the web interface:

1. **LLM (Large Language Model)**
   - **API URL**: Endpoint for the main language model (default: `http://localhost:11434/v1/chat/completions`)
   - **Model Name**: Name of the language model to use (default: `qwen3-vl:8b-thinking-q4_K_M`)

2. **UI-Model (UI Inspection Model)**
   - **API URL**: Endpoint for the UI inspection model (default: `http://localhost:2345/v1/chat/completions`)
   - **Model Name**: Name of the UI inspection model to use (default: `fara-7b`)

### API Key Configuration

API keys for the models are set through environment variables:

- `LLM_API_KEY`: API key for the main language model (default: "EMPTY")
- `UI_API_KEY`: API key for the UI-Model (default: "EMPTY")

### Additional Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `IMAGE_SERVER_HOST` | localhost | Image server hostname |
| `IMAGE_SERVER_PORT` | 12345 | Image server port |
| `RAG_DOCS_DIR` | ./docs | Documents directory for RAG |
| `RAG_INDEX_DIR` | ./index | Index storage directory |
| `MAX_REACT_ITERATIONS` | 20 | Maximum ReAct agent iterations |

## Usage

### Starting the Application

1. Run the API server:

```bash
python ops_api.py
```

2. The application will automatically open in your default web browser at `http://localhost:9000/static/index.html`

3. If the browser doesn't open automatically, navigate to `http://localhost:9000/static/index.html` manually

### Starting the UI-Model Server (Optional)

If you want to run the UI-Model locally:

```bash
python ui_model_server.py --model-path <path-to-model> --port 2345
```

### Initial Setup

1. On the web interface, configure the model settings in the "Model Configuration" section:
   - Enter the API URLs and model names for your LLM and UI-Model
   - Ensure API keys are set as environment variables if required

2. Click "Initialize Session" to create a new session with your configuration

### Using the Interface

The interface consists of four main sections:

1. **Model Configuration**: Set up your model endpoints and API keys
2. **Image Display**: View the current screen image and processed results
3. **ReAct Agent Progress**: Monitor agent execution progress and iterations
4. **Chat Interface**: Send commands and view responses

### Built-in Commands

The application supports the following built-in commands:

| Command | Description |
|---------|-------------|
| `/quit`, `/exit`, `/q` | Exit the program (close browser window manually) |
| `/clear`, `/cls` | Clear the chat history |
| `/help` | Show help information with available commands |
| `/info` | Display API status information for both models |
| `/lang [en/zh]` | Switch between English and Chinese languages |
| `/multiturn` | Enter multi-turn conversation mode (maintain context) |
| `/single` | Exit multi-turn mode, return to single-turn mode |
| `/load docs` | Load documents from the `./docs` directory and build an index |
| `/unload docs` | Disable document index (RAG functionality) |
| `/image` | Get and display the latest screen image |
| `/react [task]` | Start ReAct agent mode with specified task |
| `/stop-react` | Stop the currently running ReAct agent |

### ReAct Agent Mode

The ReAct (Reasoning + Acting) agent mode enables autonomous task execution:

1. **Starting**: Use `/react [task description]` to start the agent
2. **Progress**: Monitor progress in the ReAct Agent Progress section
3. **Stopping**: Use `/stop-react` or click "Stop Agent" button

**Approval Policies**:
- `manual`: Each dangerous operation requires approval (default)
- `auto`: All operations are automatically approved
- `strict`: All operations require approval

**Dangerous Actions** (require approval in manual mode):
- Delete, Format, Uninstall, Remove
- Erase, Wipe, Clear, Reset, Destroy
- And their Chinese equivalents

### Workflow Example

1. Start the application and initialize a session
2. Enter `/image` to capture the current screen
3. Send a command like "Click on the Settings button" to interact with UI elements
4. View the processed image with the detected UI element highlighted
5. For complex tasks, use `/react Open Settings and enable dark mode` for autonomous execution

## API Endpoints

The backend provides the following API endpoints:

### Session Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/create-session` | POST | Create a new session with configuration |
| `/status/{session_id}` | GET | Get API status information |
| `/clear-history` | POST | Clear conversation history |
| `/switch-lang` | POST | Switch language for the session |
| `/toggle-multiturn` | POST | Toggle multiturn conversation mode |

### Chat and Image

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Handle chat requests with optional image |
| `/get-image` | POST | Get the latest image from the server |

### RAG

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/build-index` | POST | Build RAG index from documents |
| `/toggle-rag` | POST | Toggle RAG functionality |

### ReAct Agent

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/react-task` | POST | Create and start async ReAct task |
| `/react-stream/{task_id}` | GET | SSE stream for task progress |
| `/react-status/{task_id}` | GET | Get task status |
| `/stop-react-task` | POST | Stop running ReAct task |
| `/approve-action/{task_id}` | POST | Approve pending action |
| `/reject-action/{task_id}` | POST | Reject pending action |
| `/set-approval-policy/{task_id}` | POST | Set approval policy |

## Response Format

The LLM should respond with structured tags for UI operations:

```xml
<action>Click</action>
<element>Settings button</element>
<reasoning>The user wants to access settings...</reasoning>
<task_status>in_progress</task_status>
```

Supported actions:
- `Click`, `Double Click`, `Right Click` - Mouse operations
- `Input` - Text input (with `<input>text</input>`)
- `Keyboard` - Keyboard shortcuts (with `<key>Ctrl+C</key>`)

## Session Management

Each session maintains its own configuration, conversation history, and state. Sessions are identified by unique session IDs generated when a new session is created. You can create multiple sessions with different configurations for different use cases.

## Language Support

The application supports both English and Chinese languages. You can switch between languages using the `/lang` command or through the API. Translation files are located in the `i18n/` directory.

## RAG (Retrieval-Augmented Generation)

The application supports RAG functionality, allowing you to build an index from documents in the `./docs` directory and use them to augment AI responses. Use the `/load docs` command to build the index and `/unload docs` to disable RAG functionality.

## UI Inspection Workflow

1. Capture the current screen image
2. Send a command to identify or interact with UI elements
3. The UI-Model processes the image and identifies the requested UI element
4. A rectangle is drawn around the detected element on the processed image
5. The processed image is displayed in the "Proposed Action" panel
6. For checkbox elements, coordinates are automatically refined

## Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Check that the API URLs are correct
   - Ensure the model servers are running
   - Verify API keys are set correctly

2. **Image Capture Issues**:
   - Ensure the image server is running on the configured host:port
   - Check permissions for accessing screen capture

3. **UI Element Detection Issues**:
   - Ensure the UI-Model is properly configured
   - Try using more specific commands to identify UI elements

4. **ReAct Agent Issues**:
   - Check the maximum iterations setting
   - Review the reasoning output for debugging
   - Use manual approval mode for safer execution

### Logging

The application logs information to `ops_api.log` and console output, which can be helpful for debugging issues. Check the logs for error messages and status updates.

## Security Considerations

- API keys are handled through environment variables, not hardcoded in the application
- CORS is enabled for development purposes, but should be restricted in production
- Sessions are stored in memory and are not persisted across server restarts
- Dangerous operations require explicit approval in manual mode

## License

MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Built with FastAPI and modern web technologies
- Supports various AI models through OpenAI-compatible API interfaces
- Designed for ease of use and extensibility
