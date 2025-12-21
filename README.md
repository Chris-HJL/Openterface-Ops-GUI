# Openterface-Ops-GUI

## Overview

Openterface-Ops-GUI is a web-based interface for interacting with AI models, specifically designed for UI inspection and interaction. It provides a seamless way to communicate with both language models and UI inspection models, enabling users to analyze and interact with UI elements through natural language commands.

## Architecture

The application consists of two main components:

1. **Backend API** (`ops_api.py`): A FastAPI server that handles requests from the frontend and communicates with AI models
2. **Frontend Interface** (`index.html`): A web-based GUI for user interaction

## Features

- **Chat Interface**: Interactive chat with AI models
- **Image Capture**: Get and display the latest screen image
- **UI Element Detection**: Identify and interact with UI elements
- **Multi-turn Conversations**: Maintain conversation context across multiple messages
- **Document Retrieval**: Build and query document indexes (RAG)
- **Language Support**: Switch between English and Chinese
- **Session Management**: Create and manage multiple sessions with different configurations

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
pip install -r requirements_ops_cli.txt
pip install -r requirements_ui_ins.txt #only when you want to deploy the UI model locally.
```

3. Set environment variables for API keys (for online model only, depending on your model setup):

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
   - **Model Name**: Name of the language model to use (default: `qwen3-vl:32b`)

2. **UI-INS (UI Inspection Model)**
   - **API URL**: Endpoint for the UI inspection model (default: `http://localhost:2345/v1/chat/completions`)
   - **Model Name**: Name of the UI inspection model to use (default: `ui-ins-7b`)

### API Key Configuration

API keys for the models are set through environment variables:

- `LLM_API_KEY`: API key for the main language model (default: "EMPTY")
- `UI_API_KEY`: API key for the UI-Ins model (default: "EMPTY")

## Usage

### Starting the Application

1. Run the API server:

```bash
python ops_api.py
```

2. The application will automatically open in your default web browser at `http://localhost:9000/static/index.html`

3. If the browser doesn't open automatically, navigate to `http://localhost:9000/static/index.html` manually

### Initial Setup

1. On the web interface, configure the model settings in the "Model Configuration" section:
   - Enter the API URLs and model names for your LLM and UI-INS models
   - Ensure API keys are set as environment variables if required

2. Click "Initialize Session" to create a new session with your configuration

### Using the Interface

The interface consists of three main sections:

1. **Model Configuration**: Set up your model endpoints and API keys
2. **Image Display**: View the current screen image and processed results
3. **Chat Interface**: Send commands and view responses

### Built-in Commands

The application supports the following built-in commands that can be entered in the chat input:

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

### Workflow Example

1. Start the application and initialize a session
2. Enter `/image` to capture the current screen
3. Send a command like "Click on the Settings button" to interact with UI elements
4. View the processed image with the detected UI element highlighted
5. Continue the conversation or enter new commands

## API Endpoints

The backend provides the following API endpoints:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/create-session` | POST | Create a new session with configuration |
| `/chat` | POST | Handle chat requests with optional image |
| `/get-image` | POST | Get the latest image from the server |
| `/build-index` | POST | Build RAG index from documents |
| `/ui-ins` | POST | UI element localization and interaction |
| `/status/{session_id}` | GET | Get API status information |
| `/switch-lang` | POST | Switch language for the session |
| `/toggle-rag` | POST | Toggle RAG functionality |
| `/toggle-multiturn` | POST | Toggle multiturn conversation mode |
| `/clear-history` | POST | Clear conversation history |

## Session Management

Each session maintains its own configuration, conversation history, and state. Sessions are identified by unique session IDs generated when a new session is created. You can create multiple sessions with different configurations for different use cases.

## Language Support

The application supports both English and Chinese languages. You can switch between languages using the `/lang` command or through the API.

## RAG (Retrieval-Augmented Generation)

The application supports RAG functionality, allowing you to build an index from documents in the `./docs` directory and use them to augment AI responses. Use the `/load docs` command to build the index and `/unload docs` to disable RAG functionality.

## UI Inspection Workflow

1. Capture the current screen image
2. Send a command to identify or interact with UI elements
3. The UI-Ins model processes the image and identifies the requested UI element
4. A rectangle is drawn around the detected element on the processed image
5. The processed image is displayed in the "Proposed Action" panel

## Troubleshooting

### Common Issues

1. **API Connection Errors**:
   - Check that the API URLs are correct
   - Ensure the model servers are running
   - Verify API keys are set correctly

2. **Image Capture Issues**:
   - Ensure the image server is running
   - Check permissions for accessing screen capture

3. **UI Element Detection Issues**:
   - Ensure the UI-Ins model is properly configured
   - Try using more specific commands to identify UI elements

### Logging

The application logs information to the console, which can be helpful for debugging issues. Check the console output for error messages and status updates.

## Security Considerations

- API keys are handled through environment variables, not hardcoded in the application
- CORS is enabled for development purposes, but should be restricted in production
- Sessions are stored in memory and are not persisted across server restarts

## License

MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## Acknowledgments

- Built with FastAPI and modern web technologies
- Supports various AI models through standard API interfaces
- Designed for ease of use and extensibility