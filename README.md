# Openterface-Ops-GUI

An AI-vision-based automated operation interface for Openterface KVM. It uses a single large language model for UI element recognition, positioning, and automated interaction, with a ReAct agent mode for complex multi-step tasks.

## Features

- **Single Model Architecture**: One VLM (Vision Language Model) handles UI understanding, element positioning, and action decisions — simplified deployment
- **ReAct Agent**: Autonomous reasoning and execution loops for complex multi-step task automation
- **Scene Adaptation**: 6 preset scenes (Auto-detect / General / BIOS / Windows / Linux / OS Installation), each with dedicated prompts
- **Rich Keyboard/Mouse Operations**: Click, double-click, triple-click, right-click, mouse move, text input, combo keys (Ctrl+C/V/A, etc.), symbol keys, scroll, lock key toggle, full-screen screenshot
- **Approval Mechanism**: Three-level approval policies (manual / auto / strict) with automatic dangerous operation detection
- **Automatic Coordinate Conversion**: Screen resolution auto-detected from screenshots — zero configuration for any resolution
- **Checkbox Precision**: OpenCV contour analysis auto-corrects click coordinates for checkbox elements
- **RAG Document Retrieval**: LlamaIndex-based vector index, supports MHTML documents
- **Multi-turn Conversations**: Context-aware multi-turn interaction mode
- **SSE Real-time Progress**: Async tasks push progress and screenshots via Server-Sent Events
- **Bilingual Support**: Frontend and backend both support English and Chinese
- **Slash Commands**: Built-in shortcuts like `/image`, `/react`, `/scene`

## Architecture

```
Openterface-Ops-GUI/
├── ops_api.py              # Application entry point (uvicorn)
├── config.py               # Global configuration
├── index.html              # Frontend single-page application
├── ui_model_server.py      # Optional: local VLM inference service
├── ops_api/                # Backend API module
│   ├── app.py              # FastAPI app factory
│   ├── endpoints.py        # API endpoint implementations
│   ├── models.py           # Pydantic model definitions
│   ├── session.py          # Session management
│   ├── task_manager.py     # ReAct async task manager
│   ├── react_context.py    # ReAct context builder
│   └── react_memory.py     # ReAct memory system
├── ops_core/               # Core functionality modules
│   ├── coord_converter.py  # Coordinate conversion (normalized ↔ pixel ↔ HID)
│   ├── api/                # API client
│   │   ├── client.py       # LLM API client (OpenAI compatible)
│   │   └── connection.py   # API connection tester
│   ├── i18n/               # Internationalization
│   │   └── translator.py   # Language translator
│   ├── image/              # Image processing
│   │   ├── encoder.py      # Base64 image encoding/decoding
│   │   └── drawer.py       # Image annotation drawing
│   ├── image_server/       # Image server client
│   │   └── client.py       # TCP image server client
│   ├── prompts/            # Prompt system
│   │   ├── types.py        # Scene type enum
│   │   ├── loader.py       # YAML prompt loader
│   │   ├── registry.py     # Prompt registry
│   │   ├── detector.py     # VLM scene detector
│   │   └── configs/        # Scene prompt configs
│   │       ├── general.yaml
│   │       ├── bios.yaml
│   │       ├── windows.yaml
│   │       ├── linux.yaml
│   │       └── os_installation.yaml
│   ├── rag/                # RAG functionality
│   │   ├── index_builder.py
│   │   ├── index_loader.py
│   │   ├── readers.py      # MHTML document reader
│   │   └── retriever.py    # Document retriever
│   ├── ui_operations/      # UI operations
│   │   ├── executor.py     # Command executor
│   │   ├── parser.py       # LLM response parser
│   │   └── checkbox_detector.py # Checkbox detector
│   └── utils/              # Utility modules
│       ├── key_map.py      # Key code mapping (with combo keys)
│       ├── text_splitter.py # Text splitter
│       └── command_sequence.py # Command sequence data structure
├── i18n/                   # Translation files (en.json / zh.json)
├── tests/                  # Test files
└── tools/                  # Development tools
```

## Quick Start

### Prerequisites

- Python 3.8+
- pip package manager

### Installation

```bash
git clone <repository-url>
cd Openterface-Ops-GUI
pip install -r requirements.txt
```

### Environment Variables

```bash
# Windows
set LLM_API_KEY=your_api_key

# Linux / macOS
export LLM_API_KEY=your_api_key
```

### Start

```bash
python ops_api.py
```

The browser opens automatically at `http://localhost:9000/static/index.html`.

### Initialization

1. Configure the LLM's API URL and model name in the "Model Configuration" panel
2. Click "Initialize Session" to create a session
3. Type `/image` to capture the current screen
4. Send a natural language command, e.g. "Click the Settings button"

## Configuration

### Model Configuration

Configure via the frontend UI:

| Setting | Default | Description |
|---------|---------|-------------|
| API URL | `http://localhost:8000/v1/chat/completions` | LLM API endpoint (OpenAI compatible) |
| Model | `qwen3.6-27b` | Model name |
| Max Iterations | `20` | Maximum ReAct iterations |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_API_KEY` | `EMPTY` | LLM API key |
| `SCREEN_WIDTH` | `1920` | Screen width (optional, auto-detect preferred) |
| `SCREEN_HEIGHT` | `1080` | Screen height (optional, auto-detect preferred) |
| `COORD_SYSTEM` | `hid` | Coordinate system (`hid` or `pixel`) |
| `SCREEN_CAPTURE_TIMEOUT` | `120` | Screenshot timeout (seconds) |

### Scene Types

| Scene | Command | Description |
|-------|---------|-------------|
| `auto` | `/scene auto` | VLM auto-detects current UI type |
| `general` | `/scene general` | General mixed scene |
| `bios` | `/scene bios` | BIOS/UEFI interface |
| `windows` | `/scene windows` | Windows desktop/applications |
| `linux` | `/scene linux` | Linux desktop/terminal |
| `os_installation` | `/scene os_installation` | OS installation interface |

## Supported UI Actions

| Action | Description |
|--------|-------------|
| `Click` | Left-click |
| `Double Click` | Double-click |
| `Right Click` | Right-click |
| `Move Mouse` | Move cursor (no click) |
| `Input` | Click + text input |
| `Keyboard` | Key press / combo key (Ctrl+C, Alt+F4, Win+E, etc.) |
| `Scroll` | Scroll wheel (up/down) |
| `Type` | Pure text input |
| `Press` | Single key press |
| `Wait` | Wait for specified time |
| `Sequence` | Multi-step operation sequence |
| `Triple Click` | Triple-click (select whole paragraph) |
| `Lock State` | Lock key toggle (CapsLock/NumLock/ScrollLock) |
| `Screenshot` | Full-screen screenshot |

### Combo Key Prefixes

| Prefix | Modifier |
|--------|----------|
| `^` | Ctrl |
| `+` | Shift |
| `!` | Alt |
| `#` | Win |

Examples: `^c` = Ctrl+C, `+!a` = Shift+Alt+A

## Built-in Commands

| Command | Description |
|---------|-------------|
| `/image` | Capture current screen |
| `/react [task]` | Start ReAct agent |
| `/stop-react` | Stop ReAct agent |
| `/scene [type]` | Switch scene type |
| `/lang [en\|zh]` | Switch language |
| `/multiturn` | Enter multi-turn mode |
| `/single` | Exit multi-turn mode |
| `/load docs` | Build RAG document index |
| `/unload docs` | Disable RAG |
| `/clear` | Clear chat history |
| `/info` | Show API status |
| `/help` | Show help |
| `/quit` | Exit program |

## ReAct Agent

The ReAct (Reasoning + Acting) mode enables the LLM to autonomously execute multi-step tasks:

### Workflow

```
User task → Capture screen → LLM reasoning → Parse action → Approval check → Execute → Result screenshot → Next iteration...
```

Each iteration:
1. Capture latest screen screenshot
2. Build enhanced prompt (with iteration history and memory)
3. Call LLM for reasoning and action decision
4. Parse `<action>`, `<element>`, `<point>`, `<reasoning>` tags from response
5. Dangerous operation detection and approval wait
6. Execute UI action
7. Capture post-action screenshot
8. Push SSE progress event

### Approval Policies

| Policy | Description |
|--------|-------------|
| `manual` | Only dangerous operations require approval (default) |
| `auto` | All operations auto-approved |
| `strict` | All operations require approval |

### Dangerous Operation Keywords

Delete, Format, Uninstall, Remove, Erase, Wipe, Clear, Reset, Destroy, and their Chinese equivalents.

## Coordinate Conversion System

Three-level coordinate conversion chain:

```
LLM normalized (0-1000) → Pixel coordinates → +Offset compensation → HID (0-4096) → TCP command
```

- **Zero Configuration**: Auto-detects resolution from screenshots
- **Y-axis Offset Compensation**: Default -10 pixels, configurable in UI
- **Boundary Clamping**: Auto-clamped to 0-4096 range

## API Endpoints

### Session Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/create-session` | POST | Create new session |
| `/status/{session_id}` | GET | Get API status |
| `/clear-history` | POST | Clear conversation history |
| `/switch-lang` | POST | Switch language |
| `/switch-scene` | POST | Switch scene type |
| `/toggle-rag` | POST | Toggle RAG |
| `/toggle-multiturn` | POST | Toggle multi-turn mode |
| `/session/{session_id}` | DELETE | Delete session |

### Chat & Image

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/chat` | POST | Handle chat request |
| `/get-image` | POST | Get latest screenshot |

### RAG

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/build-index` | POST | Build RAG index |

### ReAct Async Tasks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/react-task` | POST | Create and start async ReAct task |
| `/react-stream/{task_id}` | GET | SSE stream for task progress |
| `/react-status/{task_id}` | GET | Get task status |
| `/stop-react-task` | POST | Stop task |
| `/approve-action/{task_id}` | POST | Approve pending action |
| `/reject-action/{task_id}` | POST | Reject pending action |
| `/set-approval-policy/{task_id}` | POST | Set approval policy |
| `/approval-history/{task_id}` | GET | Get approval history |
| `/task/{task_id}` | DELETE | Delete task |

### ReAct Sync Mode

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/react` | POST | Execute ReAct synchronously (blocking) |
| `/stop-react` | POST | Stop sync ReAct |

## LLM Response Format

The LLM should return XML-tagged responses:

```xml
<action>Click</action>
<element>Settings button</element>
<reasoning>The user wants to access settings...</reasoning>
<point>500, 300</point>
<task_status>in_progress</task_status>
```

Optional tags:
- `<input>text</input>` — Input content for Input action
- `<key>Ctrl+C</key>` — Key for Keyboard action
- `<final_reasoning>...</final_reasoning>` — Final reasoning on task completion
- `<steps><step>...</step></steps>` — Steps for Sequence action

## Advanced Features

### CommandBuilder Chaining API

```python
CommandBuilder(executor)
    .click(960, 540)
    .type_text("admin")
    .press_key("Tab")
    .type_text("password123")
    .click(800, 500)
    .build()
```

### Text Splitting Strategies

Long text is auto-split into multiple Send commands. 4 strategies: `simple`, `words`, `punctuation`, `sentences`.

### Checkbox Precision Detection

When the LLM identifies a checkbox, OpenCV contour analysis searches for square contours near the returned coordinates and auto-corrects the click position.

### ReAct Memory System

Records successful and failed operation patterns, injected into subsequent iteration prompts to help the LLM avoid repeating mistakes.

## Testing

```bash
# Run tests
python -m pytest tests/

# Standalone test scripts
python test_combo_key.py    # Combo key test
python test_double_click.py  # Double-click test
python test_scroll.py        # Scroll test
```

## Development Tools

- `tools/tcpserver_simulator.py` — TCP server simulator (for development)
- `tools/tcp_test_client.py` — TCP test client
- `tools/create_test_image.py` — Create test images

## Troubleshooting

### API Connection Failed
- Verify the API URL is correct and the model server is running
- Check `LLM_API_KEY` environment variable

### Screenshot Capture Failed
- Ensure Openterface KVM device is connected
- Check `IMAGE_SERVER_HOST` and `IMAGE_SERVER_PORT` (default port: `12345`)

### Inaccurate Click Position
- Resolution is auto-detected from screenshots — usually no manual config needed
- If offset persists, configure X/Y pixel offset in the frontend
- Check `COORD_Y_OFFSET` setting (default: -10)

### ReAct Agent Stuck
- Check max iterations setting
- Review reasoning output for debugging
- Use `manual` approval mode for safer execution
- Use `/stop-react` to stop

## Logging

Logs are written to `ops_api.log` and console output.

## License

MIT License. See the LICENSE file for details.

## Contributing

Contributions are welcome! Please submit issues and pull requests.