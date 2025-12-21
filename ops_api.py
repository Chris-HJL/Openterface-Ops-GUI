#!/usr/bin/env python3
"""
Openterface Ops API Server
Provides the same functionality as ops_cli.py but via API endpoints
"""

import os
import base64
import json
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, staticfiles
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Import functions from ops_cli.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ops_cli import (
    load_translations, switch_language, encode_image_to_base64,
    setup_llamaindex, build_index_from_docs, load_index, retrieve_relevant_docs,
    test_api_connection, get_api_response, get_last_image_from_server,
    extract_click_content, draw_rectangle, parse_coordinates,
    call_ui_ins_api, process_ui_element_request
)

# Create FastAPI app
app = FastAPI(title="Openterface Ops API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/static", staticfiles.StaticFiles(directory=".", html=True), name="static")

# Session management
class Session:
    """Session class to manage per-client state"""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history = []
        self.is_multiturn_mode = False
        self.current_language = "en"
        self.current_translations = load_translations(self.current_language)
        self.index = None
        self.retriever = None
        self.rag_enabled = False
        # API configurations
        self.api_url = "http://localhost:11434/v1/chat/completions"
        self.model = "qwen3-vl:32b"
        self.ui_ins_api_url = "http://localhost:2345/v1/chat/completions"
        self.ui_ins_model = "ui-ins-7b"
        # Current image path for the session
        self.current_image_path = None

# Sessions dictionary
sessions: Dict[str, Session] = {}

# Request models
class CreateSessionRequest(BaseModel):
    language: Optional[str] = "en"
    api_url: Optional[str] = None
    model: Optional[str] = None
    ui_ins_api_url: Optional[str] = None
    ui_ins_model: Optional[str] = None

class ChatRequest(BaseModel):
    session_id: str
    prompt: str
    image: Optional[str] = None  # Base64 encoded image
    model: Optional[str] = None
    rag_enabled: Optional[bool] = None
    get_image_from_server: Optional[bool] = False

class BuildIndexRequest(BaseModel):
    session_id: str
    docs_dir: Optional[str] = "./docs"
    index_dir: Optional[str] = "./index"

class UIInsRequest(BaseModel):
    session_id: str
    image: str  # Base64 encoded image
    instruction: str

class SwitchLangRequest(BaseModel):
    session_id: str
    lang_code: str

class ToggleRequest(BaseModel):
    session_id: str
    mode: Optional[bool] = None

class ClearHistoryRequest(BaseModel):
    session_id: str

class GetImageRequest(BaseModel):
    session_id: str

# Response models
class SessionResponse(BaseModel):
    session_id: str
    message: str
    success: bool

class GetImageResponse(BaseModel):
    image: str  # Base64 encoded image
    image_id: Optional[str] = None  # Unique identifier for the image
    message: str
    success: bool

class ChatResponse(BaseModel):
    response: str
    image: Optional[str] = None  # Base64 encoded processed image
    history: List[Dict[str, Any]]
    success: bool

class IndexResponse(BaseModel):
    message: str
    success: bool

class UIInsResponse(BaseModel):
    response: str
    image: str  # Base64 encoded image with rectangle
    coordinates: Dict[str, int]
    success: bool

class StatusResponse(BaseModel):
    api_status: str
    ui_ins_status: str
    success: bool

class LangResponse(BaseModel):
    message: str
    current_lang: str
    success: bool

class ToggleResponse(BaseModel):
    message: str
    current_state: bool
    success: bool

# Helper functions
def save_base64_image(base64_str: str) -> str:
    """Save base64 encoded image to file"""
    import os
    import uuid
    from datetime import datetime
    
    # Create images directory if not exists
    os.makedirs("./images", exist_ok=True)
    
    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"api_image_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join("./images", filename)
    
    # Decode and save
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(base64_str))
    
    return filepath

def image_to_base64(filepath: str) -> str:
    """Convert image file to base64 string"""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# API endpoints
@app.post("/create-session", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create a new session"""
    import uuid
    session_id = str(uuid.uuid4())
    session = Session(session_id)
    
    # Apply custom settings if provided
    if request.language:
        session.current_language = request.language
        session.current_translations = load_translations(request.language)
    if request.api_url:
        session.api_url = request.api_url
    if request.model:
        session.model = request.model
    if request.ui_ins_api_url:
        session.ui_ins_api_url = request.ui_ins_api_url
    if request.ui_ins_model:
        session.ui_ins_model = request.ui_ins_model
    
    sessions[session_id] = session
    return SessionResponse(
        session_id=session_id,
        message="Session created successfully",
        success=True
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests"""
    # Get session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    
    # Use session or request model
    model = request.model or session.model
    rag_enabled = request.rag_enabled if request.rag_enabled is not None else session.rag_enabled
    
    # Get image path
    image_path = None
    
    # First check if we have a saved image in session
    if session.current_image_path:
        image_path = session.current_image_path
    # Then check if requested to get image from server
    elif request.get_image_from_server:
        image_path = get_last_image_from_server()
        if image_path and not image_path.startswith("./images"):
            return ChatResponse(
                response=f"Failed to get image from server: {image_path}",
                history=session.conversation_history,
                success=False
            )
    # Finally check if image was provided in request
    elif request.image:
        # Save base64 image to file
        image_path = save_base64_image(request.image)
    
    # Retrieve relevant docs if RAG enabled
    retrieved_docs = []
    if rag_enabled:
        retrieved_docs = retrieve_relevant_docs(request.prompt)
    
    # Get API response
    if session.is_multiturn_mode:
        response = get_api_response(
            request.prompt, session.api_url, model, image_path,
            session.conversation_history, retrieved_docs
        )
    else:
        response = get_api_response(
            request.prompt, session.api_url, model, image_path,
            retrieved_docs=retrieved_docs
        )
    
    # Update conversation history if in multiturn mode
    updated_history = session.conversation_history.copy()
    if session.is_multiturn_mode and response:
        # Add user message
        user_msg = {"role": "user", "content": request.prompt}
        if image_path:
            user_msg["content"] = [
                {"type": "text", "text": request.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(image_path)}"}}
            ]
        updated_history.append(user_msg)
        # Add AI response
        updated_history.append({"role": "assistant", "content": response})
        session.conversation_history = updated_history
    
    # Check for UI-Ins request
    processed_image = None
    click_content = extract_click_content(response)
    if click_content and image_path:
        # Process UI-Ins
        try:
            # Call UI-Ins API
            ui_ins_response = call_ui_ins_api(
                image_path, click_content, session.ui_ins_api_url, session.ui_ins_model
            )
            
            # Parse coordinates
            point_x, point_y = parse_coordinates(ui_ins_response)
            
            if point_x != -1:
                # Generate output image path
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join("./output", f"{name}_ui_ins{ext}")
                
                # Draw rectangle
                draw_rectangle(image_path, (point_x-20, point_y-20), (point_x+20, point_y+20), output_path)
                
                # Convert to base64
                processed_image = image_to_base64(output_path)
        except Exception as e:
            print(f"UI-Ins processing error: {str(e)}")
    
    # Clear current image path after processing
    session.current_image_path = None
    
    return ChatResponse(
        response=response,
        image=processed_image,
        history=updated_history,
        success=True
    )

@app.post("/build-index", response_model=IndexResponse)
async def build_index(request: BuildIndexRequest):
    """Build RAG index from documents"""
    # Get session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    
    # Build index
    success = build_index_from_docs(request.docs_dir, request.index_dir)
    if success:
        session.rag_enabled = True
        return IndexResponse(
            message="Index built successfully",
            success=True
        )
    else:
        return IndexResponse(
            message="Failed to build index",
            success=False
        )

@app.post("/ui-ins", response_model=UIInsResponse)
async def ui_ins(request: UIInsRequest):
    """UI element localization"""
    # Get session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    
    # Save base64 image
    image_path = save_base64_image(request.image)
    
    # Call UI-Ins API
    ui_ins_response = call_ui_ins_api(
        image_path, request.instruction, session.ui_ins_api_url, session.ui_ins_model
    )
    
    # Parse coordinates
    point_x, point_y = parse_coordinates(ui_ins_response)
    
    if point_x != -1:
        # Generate output image path
        base_name = os.path.basename(image_path)
        name, ext = os.path.splitext(base_name)
        output_path = os.path.join("./output", f"{name}_ui_ins{ext}")
        
        # Draw rectangle
        draw_rectangle(image_path, (point_x-20, point_y-20), (point_x+20, point_y+20), output_path)
        
        # Convert to base64
        processed_image = image_to_base64(output_path)
        
        return UIInsResponse(
            response=ui_ins_response,
            image=processed_image,
            coordinates={"x": point_x, "y": point_y},
            success=True
        )
    else:
        return UIInsResponse(
            response="Failed to parse coordinates",
            image="",
            coordinates={"x": -1, "y": -1},
            success=False
        )

@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str):
    """Get API status"""
    # Get session
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    
    # Check API connections
    api_status = "Available" if test_api_connection(session.api_url) else "Unavailable"
    ui_ins_status = "Available" if test_api_connection(session.ui_ins_api_url) else "Unavailable"
    
    return StatusResponse(
        api_status=api_status,
        ui_ins_status=ui_ins_status,
        success=True
    )

@app.post("/switch-lang", response_model=LangResponse)
async def switch_language_api(request: SwitchLangRequest):
    """Switch language"""
    # Get session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    
    # Switch language
    if request.lang_code in ["en", "zh"]:
        session.current_language = request.lang_code
        session.current_translations = load_translations(request.lang_code)
        return LangResponse(
            message="Language switched successfully",
            current_lang=request.lang_code,
            success=True
        )
    else:
        return LangResponse(
            message="Invalid language code",
            current_lang=session.current_language,
            success=False
        )

@app.post("/toggle-rag", response_model=ToggleResponse)
async def toggle_rag(request: ToggleRequest):
    """Toggle RAG functionality"""
    # Get session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    
    # Toggle RAG
    session.rag_enabled = not session.rag_enabled
    return ToggleResponse(
        message="RAG functionality toggled",
        current_state=session.rag_enabled,
        success=True
    )

@app.post("/toggle-multiturn", response_model=ToggleResponse)
async def toggle_multiturn(request: ToggleRequest):
    """Toggle multiturn conversation mode"""
    # Get session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    
    # Set mode if provided, otherwise toggle
    if request.mode is not None:
        session.is_multiturn_mode = request.mode
    else:
        session.is_multiturn_mode = not session.is_multiturn_mode
    
    # Clear history if switching to single turn
    if not session.is_multiturn_mode:
        session.conversation_history = []
    
    return ToggleResponse(
        message="Multiturn mode toggled",
        current_state=session.is_multiturn_mode,
        success=True
    )

@app.post("/clear-history", response_model=SessionResponse)
async def clear_history(request: ClearHistoryRequest):
    """Clear conversation history"""
    # Get session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    
    # Clear history
    session.conversation_history = []
    return SessionResponse(
        session_id=request.session_id,
        message="Conversation history cleared",
        success=True
    )

@app.post("/get-image", response_model=GetImageResponse)
async def get_image(request: GetImageRequest):
    """Get the latest image from the server"""
    # Get session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    
    # Get image from server
    image_path = get_last_image_from_server()
    if image_path and not image_path.startswith("./images"):
        return GetImageResponse(
            image="",
            message=f"Failed to get image from server: {image_path}",
            success=False
        )
    
    # Save image path to session
    session.current_image_path = image_path
    
    # Convert image to base64
    try:
        image_base64 = image_to_base64(image_path)
        return GetImageResponse(
            image=image_base64,
            message="Image retrieved successfully",
            success=True
        )
    except Exception as e:
        return GetImageResponse(
            image="",
            message=f"Failed to process image: {str(e)}",
            success=False
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Openterface Ops API Server",
        "version": "1.0.0",
        "endpoints": [
            "/create-session",
            "/chat",
            "/get-image",
            "/build-index",
            "/ui-ins",
            "/status/{session_id}",
            "/switch-lang",
            "/toggle-rag",
            "/toggle-multiturn",
            "/clear-history"
        ]
    }

if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import time
    import threading
    
    # 启动后自动打开浏览器
    def open_browser():
        time.sleep(1)  # 等待服务器启动
        webbrowser.open("http://localhost:9000/static/index.html")
    
    # 在新线程中打开浏览器
    threading.Thread(target=open_browser).start()
    
    uvicorn.run(app, host="0.0.0.0", port=9000)
