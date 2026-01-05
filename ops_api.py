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
from PIL import Image

# Import functions from ops_cli.py
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from ops_cli import (
    load_translations, switch_language, encode_image_to_base64,
    setup_llamaindex, build_index_from_docs, load_index, retrieve_relevant_docs,
    test_api_connection, get_api_response, get_last_image_from_server,
    extract_click_content, draw_rectangle, parse_coordinates,
    call_ui_ins_api, process_ui_element_request, send_script_command
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
        self.model = "qwen3-vl:8b-thinking-q4_K_M"
        self.ui_model_api_url = "http://localhost:2345/v1/chat/completions"
        self.ui_model = "fara-7b"
        # Current image path for the session
        self.current_image_path = None
        # ReAct agent configurations
        self.react_enabled = False
        self.react_max_iterations = 20
        self.react_current_iteration = 0
        self.react_task_description = None
        self.react_is_running = False

# Sessions dictionary
sessions: Dict[str, Session] = {}

# Request models
class CreateSessionRequest(BaseModel):
    language: Optional[str] = "en"
    api_url: Optional[str] = None
    model: Optional[str] = None
    ui_model_api_url: Optional[str] = None
    ui_model: Optional[str] = None

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

class ReactRequest(BaseModel):
    session_id: str
    task: str
    max_iterations: Optional[int] = 20
    rag_enabled: Optional[bool] = None
    model: Optional[str] = None

class StopReactRequest(BaseModel):
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

class StatusResponse(BaseModel):
    api_status: str
    ui_model_status: str
    success: bool

class LangResponse(BaseModel):
    message: str
    current_lang: str
    success: bool

class ToggleResponse(BaseModel):
    message: str
    current_state: bool
    success: bool

class ReactResponse(BaseModel):
    message: str
    iterations_completed: int
    final_status: str
    success: bool
    images: Optional[List[str]] = None  # List of base64 images from each iteration

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

def extract_action_and_element(text: str) -> tuple:
    """
    Extract action and element from response
    
    Args:
        text: Response text containing <action> and <element> tags
        
    Returns:
        Tuple of (action, element, input_content, key_content) where each can be None if not found
    """
    import re
    
    # Extract action
    action_pattern = r'<action>(.*?)</action>'
    action_matches = re.findall(action_pattern, text, re.DOTALL)
    action = action_matches[0].strip() if action_matches else None
    
    # Extract element
    element_pattern = r'<element>(.*?)</element>'
    element_matches = re.findall(element_pattern, text, re.DOTALL)
    element = element_matches[0].strip() if element_matches else None
    
    # Extract input content
    input_pattern = r'<input>(.*?)</input>'
    input_matches = re.findall(input_pattern, text, re.DOTALL)
    input_content = input_matches[0].strip() if input_matches else None
    
    # Extract key content
    key_pattern = r'<key>(.*?)</key>'
    key_matches = re.findall(key_pattern, text, re.DOTALL)
    key_content = key_matches[0].strip() if key_matches else None
    
    return (action, element, input_content, key_content)

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
    if request.ui_model_api_url:
        session.ui_model_api_url = request.ui_model_api_url
    if request.ui_model:
        session.ui_model = request.ui_model
    
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
    print(f"[Session {request.session_id}] API response: {response}")
    
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
    
    # Check for UI-Model request
    processed_image = None
    action, element, input_content, key_content = extract_action_and_element(response)
    
    if action in ["Click", "Double Click", "Right Click"] and element and image_path:
        # Process UI-Model
        try:
            # Call UI-Model API with element content
            ui_model_response = call_ui_ins_api(
                image_path, element, session.ui_model_api_url, session.ui_model
            )

            print(f"[Session {request.session_id}] UI-Model response: {ui_model_response}")
            
            # Parse coordinates
            point_x, point_y = parse_coordinates(ui_model_response)
            
            if point_x != -1:
                # Generate output image path
                base_name = os.path.basename(image_path)
                name, ext = os.path.splitext(base_name)
                output_path = os.path.join("./output", f"{name}_ui_model{ext}")

                # Default box size
                box_size = 50
                left = point_x - box_size // 2
                top = point_y - box_size // 2
                right = point_x + box_size // 2
                bottom = point_y + box_size // 2

                # Ensure box is within image bounds
                width, height = Image.open(image_path).size
                left = max(0, left)
                top = max(0, top)
                right = min(width - 1, right)
                bottom = min(height - 1, bottom)
                
                # Draw rectangle
                draw_rectangle(image_path, (left, top), (right, bottom), output_path)   
                
                # Convert to base64
                processed_image = image_to_base64(output_path)

                # Build and send script command to server
                if action == "Click":
                    script_command = f'Send "{{Click {point_x}, {point_y}}}"'
                elif action == "Double Click":
                    script_command = f'Send "{{Click {point_x}, {point_y}}}"\nSend "{{Click {point_x}, {point_y}}}"'
                elif action == "Right Click":
                    script_command = f'Send "{{Click {point_x}, {point_y} Right}}"'
                else:
                    script_command = None
                
                if script_command:
                    send_script_command(script_command)
        except Exception as e:
            print(f"UI-Model processing error: {str(e)}")
    elif action == "Input" and input_content:
        # Process Input action - send text directly
        script_command = f'Send "{input_content}"'
        send_script_command(script_command)
        print(f"[Session {request.session_id}] Executed Input: {input_content}")
    elif action == "Keyboard" and key_content:
        # Process Keyboard action - send key command
        script_command = f'Send "{{{key_content}}}"'
        send_script_command(script_command)
        print(f"[Session {request.session_id}] Executed Keyboard: {key_content}")
    
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

@app.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str):
    """Get API status"""
    # Get session
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]
    
    # Check API connections
    api_status = "Available" if test_api_connection(session.api_url) else "Unavailable"
    ui_model_status = "Available" if test_api_connection(session.ui_model_api_url) else "Unavailable"
    
    return StatusResponse(
        api_status=api_status,
        ui_model_status=ui_model_status,
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

@app.post("/react", response_model=ReactResponse)
async def react(request: ReactRequest):
    """Start ReAct agent mode to autonomously complete a task"""
    # Get session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    
    # Check if already running
    if session.react_is_running:
        return ReactResponse(
            message="ReAct agent is already running",
            iterations_completed=0,
            final_status="already_running",
            success=False
        )
    
    # Initialize ReAct mode
    session.react_enabled = True
    session.react_max_iterations = request.max_iterations
    session.react_current_iteration = 0
    session.react_task_description = request.task
    session.react_is_running = True
    
    # Use session or request model and RAG setting
    model = request.model or session.model
    rag_enabled = request.rag_enabled if request.rag_enabled is not None else session.rag_enabled
    
    # Store images from each iteration
    iteration_images = []
    
    try:
        while session.react_is_running and session.react_current_iteration < session.react_max_iterations:
            session.react_current_iteration += 1
            iteration_num = session.react_current_iteration
            
            print(f"[ReAct Session {request.session_id}] Iteration {iteration_num}/{session.react_max_iterations}")
            
            # Get latest image from server
            image_path = get_last_image_from_server()
            if image_path and not image_path.startswith("./images"):
                return ReactResponse(
                    message=f"Failed to get image from server: {image_path}",
                    iterations_completed=iteration_num - 1,
                    final_status="error",
                    success=False,
                    images=iteration_images
                )
            
            # Build prompt for task completion check
            check_prompt = f"""Task: {session.react_task_description}

                Current iteration: {iteration_num}/{session.react_max_iterations}

                Please analyze the current screen and determine if the task has been completed.
                Respond with one of the following:
                - <task_status>completed</task_status> if the task is done
                - <task_status>in_progress</task_status> if the task is not yet completed or the screen may still be loading

                If the screen may still be loading, no other information is needed.
                Else if not completed, also provide:
                - <action>Click</action> or <action>Double Click</action> or <action>none</action>
                - <element>description of UI element to interact with</element> (if action is Click or Double Click)
                - <reasoning>brief explanation of what needs to be done next</reasoning>
                Else if completed, also provide:
                - <final_reasoning>brief explanation of the task completion</final_reasoning>
            """
            
            # Retrieve relevant docs if RAG enabled
            retrieved_docs = []
            if rag_enabled:
                retrieved_docs = retrieve_relevant_docs(session.react_task_description)
            
            # Get LLM response for task completion check
            response = get_api_response(
                check_prompt, session.api_url, model, image_path,
                retrieved_docs=retrieved_docs
            )
            print(f"[ReAct Session {request.session_id}] LLM response: {response}")
            
            # Check if task is completed
            task_status_pattern = r'<task_status>(.*?)</task_status>'
            task_status_matches = [m.strip() for m in __import__('re').findall(task_status_pattern, response, __import__('re').DOTALL)]
            task_status = task_status_matches[0] if task_status_matches else "in_progress"
            
            if task_status == "completed":
                print(f"[ReAct Session {request.session_id}] Task completed at iteration {iteration_num}")
                session.react_is_running = False
                
                # Extract final_reasoning from response
                final_reasoning_pattern = r'<final_reasoning>(.*?)</final_reasoning>'
                final_reasoning_matches = [m.strip() for m in __import__('re').findall(final_reasoning_pattern, response, __import__('re').DOTALL)]
                final_reasoning = final_reasoning_matches[0] if final_reasoning_matches else None
                
                # Build message with final_reasoning
                message = f"Task completed successfully in {iteration_num} iterations"
                if final_reasoning:
                    message += f"\n<final_reasoning>{final_reasoning}</final_reasoning>"
                
                return ReactResponse(
                    message=message,
                    iterations_completed=iteration_num,
                    final_status="completed",
                    success=True,
                    images=iteration_images
                )
            
            # Extract action and element for next step
            action, element, input_content, key_content = extract_action_and_element(response)
            
            # Process UI-Model and execute action if needed
            processed_image = None
            if action in ["Click", "Double Click", "Right Click"] and element and image_path:
                try:
                    # Call UI-Model API with element content
                    ui_model_response = call_ui_ins_api(
                        image_path, element, session.ui_model_api_url, session.ui_model
                    )
                    print(f"[Session {request.session_id}] UI-Model response: {ui_model_response}")

                    
                    # Parse coordinates
                    point_x, point_y = parse_coordinates(ui_model_response)
                    
                    if point_x != -1:
                        # Generate output image path
                        base_name = os.path.basename(image_path)
                        name, ext = os.path.splitext(base_name)
                        output_path = os.path.join("./output", f"{name}_react_iter{iteration_num}{ext}")

                        # Default box size
                        box_size = 50
                        left = point_x - box_size // 2
                        top = point_y - box_size // 2
                        right = point_x + box_size // 2
                        bottom = point_y + box_size // 2

                        # Ensure box is within image bounds
                        width, height = Image.open(image_path).size
                        left = max(0, left)
                        top = max(0, top)
                        right = min(width - 1, right)
                        bottom = min(height - 1, bottom)
                        
                        # Draw rectangle
                        draw_rectangle(image_path, (left, top), (right, bottom), output_path)   
                        
                        # Convert to base64 and store
                        processed_image = image_to_base64(output_path)
                        iteration_images.append(processed_image)

                        # Build and send script command to server
                        if action == "Click":
                            script_command = f'Send "{{Click {point_x}, {point_y}}}"'
                        elif action == "Double Click":
                            script_command = f'Send "{{Click {point_x}, {point_y}}}"\nSend "{{Click {point_x}, {point_y}}}"'
                        elif action == "Right Click":
                            script_command = f'Send "{{Click {point_x}, {point_y} Right}}"'
                        else:
                            script_command = None
                        
                        if script_command:
                            send_script_command(script_command)
                            print(f"[ReAct Session {request.session_id}] Executed {action} at ({point_x}, {point_y})")
                except Exception as e:
                    print(f"[ReAct Session {request.session_id}] UI-Model processing error: {str(e)}")
            elif action == "Input" and input_content:
                # Process Input action - send text directly
                script_command = f'Send "{input_content}"'
                send_script_command(script_command)
                print(f"[ReAct Session {request.session_id}] Executed Input: {input_content}")
            elif action == "Keyboard" and key_content:
                # Process Keyboard action - send key command
                script_command = f'Send "{{{key_content}}}"'
                send_script_command(script_command)
                print(f"[ReAct Session {request.session_id}] Executed Keyboard: {key_content}")
            
            # Wait 3 seconds before next iteration
            import time
            time.sleep(3)
        
        # Max iterations reached
        session.react_is_running = False
        return ReactResponse(
            message=f"Reached maximum iterations ({session.react_max_iterations}) without completing task",
            iterations_completed=session.react_max_iterations,
            final_status="max_iterations_reached",
            success=False,
            images=iteration_images
        )
    
    except Exception as e:
        session.react_is_running = False
        print(f"[ReAct Session {request.session_id}] Error: {str(e)}")
        return ReactResponse(
            message=f"ReAct agent encountered an error: {str(e)}",
            iterations_completed=session.react_current_iteration,
            final_status="error",
            success=False,
            images=iteration_images
        )

@app.post("/stop-react", response_model=SessionResponse)
async def stop_react(request: StopReactRequest):
    """Stop the running ReAct agent"""
    # Get session
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]
    
    # Stop ReAct agent
    if session.react_is_running:
        session.react_is_running = False
        return SessionResponse(
            session_id=request.session_id,
            message="ReAct agent stopped successfully",
            success=True
        )
    else:
        return SessionResponse(
            session_id=request.session_id,
            message="ReAct agent is not running",
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
            "/status/{session_id}",
            "/switch-lang",
            "/toggle-rag",
            "/toggle-multiturn",
            "/clear-history",
            "/react",
            "/stop-react"
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
