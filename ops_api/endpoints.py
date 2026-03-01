"""
API endpoints implementation
"""
import os
import base64
import uuid
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict
from fastapi import HTTPException
from PIL import Image

from .models import *
from .session import Session
from .react_memory import IterationRecord
from .react_context import ReActContextBuilder
from .task_manager import task_manager
from ops_core import (
    Translator,
    ImageEncoder,
    ImageDrawer,
    IndexBuilder,
    DocumentRetriever,
    APIConnectionTester,
    ResponseParser
)
from config import Config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ops_api.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SessionManager:
    """Session manager with cleanup support"""

    def __init__(self):
        self.sessions: Dict[str, Session] = {}
        self.lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = True

    def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("[SessionManager] Started background cleanup task")

    def stop_cleanup_task(self):
        """Stop background cleanup task"""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("[SessionManager] Stopped background cleanup task")

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                await asyncio.sleep(Config.CLEANUP_INTERVAL_SECONDS)
                await self.cleanup_expired_sessions()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[SessionManager] Cleanup loop error: {e}")

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        now = datetime.now()
        ttl = timedelta(seconds=Config.SESSION_TTL_SECONDS)
        expired_session_ids = []

        async with self.lock:
            for session_id, session in list(self.sessions.items()):
                if now - session.updated_at > ttl:
                    expired_session_ids.append(session_id)

            for session_id in expired_session_ids:
                del self.sessions[session_id]
                logger.info(f"[SessionManager] Cleaned up expired session {session_id}")

        if expired_session_ids:
            logger.info(f"[SessionManager] Cleaned up {len(expired_session_ids)} expired sessions")

    async def create_session(self, request: "CreateSessionRequest") -> tuple:
        """Create new session with cleanup check"""
        async with self.lock:
            if len(self.sessions) >= Config.MAX_SESSIONS:
                self._evict_oldest_session()

            session_id = str(uuid.uuid4())
            session = Session(session_id)

            if request.language:
                session.switch_language(request.language)
            if request.api_url:
                session.api_url = request.api_url
            if request.model:
                session.model = request.model
            if request.ui_model_api_url:
                session.ui_model_api_url = request.ui_model_api_url
            if request.ui_model:
                session.ui_model = request.ui_model

            self.sessions[session_id] = session

        logger.info(f"[SessionManager] Created session {session_id}")
        return session_id, session

    def _evict_oldest_session(self):
        """Evict the oldest session when limit is reached (must be called with lock held)"""
        if self.sessions:
            oldest_session_id = min(
                self.sessions.keys(),
                key=lambda sid: self.sessions[sid].updated_at
            )
            del self.sessions[oldest_session_id]
            logger.info(f"[SessionManager] Evicted oldest session {oldest_session_id} to make room")

    def get_session(self, session_id: str) -> Optional[Session]:
        """Get session and update activity timestamp"""
        session = self.sessions.get(session_id)
        if session:
            session.touch()
        return session

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session manually"""
        async with self.lock:
            if session_id in self.sessions:
                del self.sessions[session_id]
                logger.info(f"[SessionManager] Manually deleted session {session_id}")
                return True
        return False

session_manager = SessionManager()

# Create FastAPI application
from fastapi import APIRouter
router = APIRouter()

# Helper functions
def save_base64_image(base64_str: str) -> str:
    """Save base64 encoded image to file"""
    # Create images directory
    os.makedirs(Config.IMAGES_DIR, exist_ok=True)

    # Generate filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"api_image_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(Config.IMAGES_DIR, filename)

    # Decode and save
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(base64_str))

    return filepath

def image_to_base64(filepath: str) -> str:
    """Convert image file to base64 string"""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# API endpoints
@router.post("/create-session", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """Create new session"""
    session_id, session = await session_manager.create_session(request)
    return SessionResponse(
        session_id=session_id,
        message="Session created successfully",
        success=True
    )

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Process chat request"""
    logger.info(f"[Chat Request] Session: {request.session_id}, Prompt: {request.prompt[:100]}...")

    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        logger.error(f"[Chat] Session not found: {request.session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    # Use session or requested model
    model = request.model or session.model
    rag_enabled = request.rag_enabled if request.rag_enabled is not None else session.rag_enabled

    logger.info(f"[Chat] Model: {model}, RAG enabled: {rag_enabled}, Multiturn mode: {session.is_multiturn_mode}")

    # Get image path
    image_path = None

    # First check if session has saved image
    if session.current_image_path:
        image_path = session.current_image_path
        logger.info(f"[Chat] Using session image: {image_path}")
    # Then check if request to get image from server
    elif request.get_image_from_server:
        logger.info("[Chat] Fetching image from server...")
        image_server_client = session.image_server_client

        # 使用 gettargetscreen 命令获取屏幕图像
        image_path = image_server_client.get_target_screen()

        # 错误检查
        if not image_path or image_path.startswith("Error:") or not image_path.startswith("./images"):
            logger.error(f"[Chat] Failed to get image from server: {image_path}")
            return ChatResponse(
                response=f"Failed to get image from server: {image_path}",
                history=session.conversation_history,
                success=False
            )
        logger.info(f"[Chat] Got image from server: {image_path}")
    # Finally check if image provided in request
    elif request.image:
        # Save base64 image to file
        logger.info("[Chat] Saving base64 image to file...")
        image_path = save_base64_image(request.image)
        logger.info(f"[Chat] Saved image to: {image_path}")

    # If RAG enabled, retrieve relevant documents
    retrieved_docs = []
    if rag_enabled and session.retriever:
        logger.info("[Chat] Retrieving relevant documents...")
        retrieved_docs = session.retriever.retrieve(request.prompt)
        logger.info(f"[Chat] Retrieved {len(retrieved_docs)} documents")

    # Get API response
    logger.info(f"[Chat] Calling LLM API at {session.api_url}...")
    api_client = session.get_llm_api_client(model=model)
    if session.is_multiturn_mode:
        response = api_client.get_response(
            request.prompt,
            image_path=image_path,
            history=session.conversation_history,
            retrieved_docs=retrieved_docs
        )
    else:
        response = api_client.get_response(
            request.prompt,
            image_path=image_path,
            retrieved_docs=retrieved_docs
        )
    
    logger.info(f"[Chat] LLM Response: {response[:200]}...")

    # Update conversation history (if multiturn mode)
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

    # Check UI-Model request
    processed_image = None
    parser = ResponseParser()
    action, element, input_content, key_content = parser.extract_action_and_element(response)
    
    logger.info(f"[Chat] Parsed action: {action}, element: {element}, input: {input_content}, key: {key_content}")

    if action in ["Click", "Double Click", "Right Click"] and element and image_path:
        # Process UI-Model
        logger.info(f"[Chat] Processing UI-Model request: {action} on {element}")
        try:
            executor = session.get_command_executor()
            logger.info(f"[Chat] Calling UI-Model API at {session.ui_model_api_url} with model {session.ui_model}")
            success, result = executor.process_ui_element_request(
                image_path, element, action, element
            )

            if success:
                logger.info(f"[Chat] UI-Model success: {result}")
                if isinstance(result, str) and os.path.exists(result):
                    # Convert to base64
                    processed_image = image_to_base64(result)
                    logger.info(f"[Chat] Processed image generated: {result}")
            else:
                logger.error(f"[Chat] UI-Model failed: {result}")

        except Exception as e:
            logger.error(f"[Chat] UI-Model processing error: {str(e)}", exc_info=True)
    elif action == "Input" and input_content:
        # Process Input action - send text directly
        logger.info(f"[Chat] Sending input: {input_content}")
        image_server_client = session.image_server_client
        script_command = f'Send "{input_content}"'
        image_server_client.send_script_command(script_command)
        logger.info(f"[Chat] Input sent successfully")
    elif action == "Keyboard" and key_content:
        # Process Keyboard action - send key command
        logger.info(f"[Chat] Sending keyboard key: {key_content}")
        image_server_client = session.image_server_client
        script_command = f'Send "{{{key_content}}}"'
        image_server_client.send_script_command(script_command)
        logger.info(f"[Chat] Keyboard command sent successfully")

    # Clear current image path after processing
    session.current_image_path = None

    return ChatResponse(
        response=response,
        image=processed_image,
        history=updated_history,
        success=True
    )

@router.post("/build-index", response_model=IndexResponse)
async def build_index(request: BuildIndexRequest):
    """Build RAG index from documents"""
    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Build index
    builder = IndexBuilder()
    success = builder.build_index(request.docs_dir, request.index_dir)
    if success:
        session.rag_enabled = True
        session.retriever = DocumentRetriever(request.index_dir)
        return IndexResponse(
            message="Index built successfully",
            success=True
        )
    else:
        return IndexResponse(
            message="Failed to build index",
            success=False
        )

# New async task endpoint
@router.post("/react-task", response_model=CreateReactTaskResponse)
async def create_react_task(request: CreateReactTaskRequest):
    """Create and start async ReAct task"""
    logger.info(f"[ReAct Task] Creating task for session {request.session_id}")
    logger.info(f"[ReAct Task] Task: {request.task}, Max iterations: {request.max_iterations}, Approval policy: {request.approval_policy}")

    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        logger.error(f"[ReAct Task] Session not found: {request.session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    # Apply custom settings
    if request.model:
        session.model = request.model
    if request.rag_enabled is not None:
        session.rag_enabled = request.rag_enabled

    # Create task
    task_id = await task_manager.create_task(
        session_id=request.session_id,
        task_description=request.task,
        max_iterations=request.max_iterations,
        session=session,
        approval_policy=request.approval_policy
    )

    # Start task
    await task_manager.start_task(task_id)

    logger.info(f"[ReAct Task] Task {task_id} created and started")

    return CreateReactTaskResponse(
        task_id=task_id,
        message="ReAct task created and started",
        success=True
    )


@router.get("/react-stream/{task_id}")
async def react_stream(task_id: str):
    """SSE stream task progress"""
    from fastapi.responses import StreamingResponse
    import json

    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        try:
            while True:
                # Check task status
                if task.status == "completed":
                    logger.info(f"[ReAct Stream] Task completed, sending completed event")
                    data = {
                        "event": "completed",
                        "data": {
                            "iterations_completed": task.current_iteration,
                            "final_status": task.final_status,
                            "message": task.message,
                            "images": task.iteration_images
                        }
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    break
                elif task.status == "stopped":
                    logger.info(f"[ReAct Stream] Task stopped, sending stopped event. Iterations: {task.current_iteration}, Task ID: {task.task_id}")
                    data = {
                        "event": "stopped",
                        "data": {
                            "iterations_completed": task.current_iteration
                        }
                    }
                    event_json = json.dumps(data)
                    logger.info(f"[ReAct Stream] Sending event JSON: {event_json}")
                    yield f"data: {event_json}\n\n"
                    logger.info(f"[ReAct Stream] Stopped event sent successfully")
                    break
                elif task.status == "error":
                    logger.info(f"[ReAct Stream] Task error, sending error event")
                    data = {
                        "event": "error",
                        "data": {
                            "error": task.error_message
                        }
                    }
                    yield f"data: {json.dumps(data)}\n\n"
                    break

                # Get updates from event queue
                try:
                    event = await asyncio.wait_for(task.event_queue.get(), timeout=0.5)
                    if event["type"] == "progress":
                        progress_data = event["data"]
                        data = {
                            "event": "progress",
                            "data": {
                                "iteration": task.current_iteration,
                                "max_iterations": task.max_iterations,
                                "status": progress_data.get("status"),
                                "action": progress_data.get("action"),
                                "element": progress_data.get("element"),
                                "key_content": progress_data.get("key_content"),
                                "reasoning": progress_data.get("reasoning"),
                                "task_status": progress_data.get("task_status"),
                                "image_path": progress_data.get("image_path")
                            }
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                    elif event["type"] == "approval_required":
                        data = {
                            "event": "approval_required",
                            "data": event["data"]
                        }
                        yield f"data: {json.dumps(data)}\n\n"
                except asyncio.TimeoutError:
                    # Timeout continue loop
                    continue

        except Exception as e:
            logger.error(f"[ReAct Stream] Error in event generator: {str(e)}", exc_info=True)
            data = {
                "event": "error",
                "data": {
                    "error": str(e)
                }
            }
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@router.get("/react-status/{task_id}", response_model=ReactTaskStatusResponse)
async def get_react_status(task_id: str):
    """Get task status"""
    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return ReactTaskStatusResponse(
        task_id=task_id,
        status=task.status,
        current_iteration=task.current_iteration,
        max_iterations=task.max_iterations,
        last_action=task.last_action,
        last_element=task.last_element,
        last_reasoning=task.last_reasoning,
        last_task_status=task.last_task_status,
        last_status=task.last_status,
        created_at=task.created_at.isoformat(),
        updated_at=task.updated_at.isoformat(),
        approval_policy=task.approval_policy,
        pending_approval=task.pending_approval
    )


@router.post("/stop-react-task")
async def stop_react_task(request: StopReactTaskRequest):
    """Stop async ReAct task"""
    logger.info(f"[ReAct Task] Stopping task {request.task_id}")

    task = task_manager.get_task(request.task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    await task_manager.stop_task(request.task_id)

    return {
        "message": "Task stop signal sent",
        "task_id": request.task_id,
        "success": True
    }

@router.post("/approve-action/{task_id}", response_model=ApprovalResponse)
async def approve_action(task_id: str, request: ApprovalRequest):
    """Approve action"""
    logger.info(f"[ReAct Task] Approving action for task {task_id}")

    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    success = await task_manager.approve_action(task_id)
    if success:
        return ApprovalResponse(
            success=True,
            message="Action approved",
            task_id=task_id,
            iteration=task.current_iteration
        )
    else:
        return ApprovalResponse(
            success=False,
            message="Failed to approve action - task not in waiting_approval state",
            task_id=task_id
        )

@router.post("/reject-action/{task_id}", response_model=ApprovalResponse)
async def reject_action(task_id: str, request: RejectionRequest):
    """Reject action"""
    logger.info(f"[ReAct Task] Rejecting action for task {task_id}: {request.reason}")

    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    success = await task_manager.reject_action(task_id, request.reason)
    if success:
        return ApprovalResponse(
            success=True,
            message="Action rejected",
            task_id=task_id,
            iteration=task.current_iteration
        )
    else:
        return ApprovalResponse(
            success=False,
            message="Failed to reject action - task not in waiting_approval state",
            task_id=task_id
        )

@router.post("/set-approval-policy/{task_id}")
async def set_approval_policy(task_id: str, policy: str):
    """Set approval policy"""
    logger.info(f"[ReAct Task] Setting approval policy for task {task_id} to {policy}")

    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if policy not in ["auto", "manual", "strict"]:
        raise HTTPException(status_code=400, detail="Invalid policy. Must be 'auto', 'manual', or 'strict'")

    task_manager.set_approval_policy(task_id, policy)

    return {
        "message": f"Approval policy set to {policy}",
        "task_id": task_id,
        "success": True
    }

@router.get("/approval-history/{task_id}", response_model=ApprovalHistoryResponse)
async def get_approval_history(task_id: str):
    """Get approval history"""
    logger.info(f"[ReAct Task] Getting approval history for task {task_id}")

    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    return ApprovalHistoryResponse(
        task_id=task_id,
        approval_policy=task.approval_policy,
        approval_history=task.approval_history,
        success=True
    )

@router.get("/status/{session_id}", response_model=StatusResponse)
async def get_status(session_id: str):
    """Get API status"""
    # Get session
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check API connection
    tester = APIConnectionTester()
    api_status = "Available" if tester.test_connection(session.api_url) else "Unavailable"
    ui_model_status = "Available" if tester.test_connection(session.ui_model_api_url) else "Unavailable"

    return StatusResponse(
        api_status=api_status,
        ui_model_status=ui_model_status,
        success=True
    )

@router.post("/switch-lang", response_model=LangResponse)
async def switch_language_api(request: SwitchLangRequest):
    """Switch language"""
    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Switch language
    if request.lang_code in ["en", "zh"]:
        if session.switch_language(request.lang_code):
            return LangResponse(
                message="Language switched successfully",
                current_lang=request.lang_code,
                success=True
            )
    return LangResponse(
        message="Invalid language code",
        current_lang=session.current_language,
        success=False
    )

@router.post("/switch-scene", response_model=SceneResponse)
async def switch_scene_api(request: SwitchSceneRequest):
    """Switch scene type for prompt selection"""
    from ops_core.prompts import SceneType
    
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    valid_scenes = [s.value for s in SceneType]
    if request.scene_type.lower() in valid_scenes:
        if session.switch_scene(request.scene_type):
            logger.info(f"[Scene] Session {request.session_id}: Scene switched to {request.scene_type.lower()}")
            return SceneResponse(
                message="Scene switched successfully",
                current_scene=request.scene_type.lower(),
                success=True
            )
    logger.warning(f"[Scene] Session {request.session_id}: Invalid scene type '{request.scene_type}'")
    return SceneResponse(
        message=f"Invalid scene type. Valid options: {valid_scenes}",
        current_scene=session.scene_type.value,
        success=False
    )

@router.post("/toggle-rag", response_model=ToggleResponse)
async def toggle_rag(request: ToggleRequest):
    """Toggle RAG functionality"""
    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Toggle RAG
    session.rag_enabled = not session.rag_enabled
    return ToggleResponse(
        message="RAG functionality toggled",
        current_state=session.rag_enabled,
        success=True
    )

@router.post("/toggle-multiturn", response_model=ToggleResponse)
async def toggle_multiturn(request: ToggleRequest):
    """Toggle multiturn mode"""
    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # If mode provided then set, otherwise toggle
    if request.mode is not None:
        session.is_multiturn_mode = request.mode
    else:
        session.is_multiturn_mode = not session.is_multiturn_mode

    # If switching to single turn mode, clear history
    if not session.is_multiturn_mode:
        session.clear_history()

    return ToggleResponse(
        message="Multiturn mode toggled",
        current_state=session.is_multiturn_mode,
        success=True
    )

@router.post("/clear-history", response_model=SessionResponse)
async def clear_history(request: ClearHistoryRequest):
    """Clear conversation history"""
    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Clear history
    session.clear_history()
    return SessionResponse(
        session_id=request.session_id,
        message="Conversation history cleared",
        success=True
    )

@router.post("/get-image", response_model=GetImageResponse)
async def get_image(request: GetImageRequest):
    """Get latest image from server"""
    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Get image from server using cached client
    image_server_client = session.image_server_client

    # 使用 gettargetscreen 命令获取屏幕图像
    image_path = image_server_client.get_target_screen()

    # 错误检查
    if not image_path or image_path.startswith("Error:") or not image_path.startswith("./images"):
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

@router.post("/react", response_model=ReactResponse)
async def react(request: ReactRequest):
    """Start ReAct agent mode to complete task independently"""
    logger.info(f"[ReAct] Starting ReAct agent for session {request.session_id}")
    logger.info(f"[ReAct] Task: {request.task}, Max iterations: {request.max_iterations}")
    
    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        logger.error(f"[ReAct] Session not found: {request.session_id}")
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if already running
    if session.react_is_running:
        logger.warning(f"[ReAct] Agent already running for session {request.session_id}")
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
    # Clear cached detected scene for new task
    session.react_detected_scene = None

    # Initialize memory
    session.initialize_react_memory(request.task)
    context_builder = ReActContextBuilder(session.react_memory, max_context_iterations=5)

    # Use session or requested model and RAG settings
    model = request.model or session.model
    rag_enabled = request.rag_enabled if request.rag_enabled is not None else session.rag_enabled

    logger.info(f"[ReAct] Model: {model}, RAG enabled: {rag_enabled}")

    # Store image for each iteration
    iteration_images = []

    try:
        while session.react_is_running and session.react_current_iteration < session.react_max_iterations:
            session.react_current_iteration += 1
            iteration_num = session.react_current_iteration

            logger.info(f"[ReAct Session {request.session_id}] Iteration {iteration_num}/{session.react_max_iterations}")

            # Fetch latest image from server using cached client
            logger.info(f"[ReAct] Fetching image from server for iteration {iteration_num}")
            image_server_client = session.image_server_client

            # 使用 gettargetscreen 命令获取屏幕图像
            image_path = image_server_client.get_target_screen()

            # 错误检查
            if not image_path or image_path.startswith("Error:") or not image_path.startswith("./images"):
                logger.error(f"[ReAct] Failed to get image from server: {image_path}")

                # Log failed iteration
                iteration_record = IterationRecord(
                    iteration_number=iteration_num,
                    timestamp=datetime.now().isoformat(),
                    image_path=image_path,
                    prompt="",
                    llm_response="",
                    execution_success=False,
                    execution_error=f"Failed to get image from server: {image_path}",
                    task_status="error"
                )
                session.react_memory_store.add_iteration(session.session_id, iteration_record)

                return ReactResponse(
                    message=f"Failed to get image from server: {image_path}",
                    iterations_completed=iteration_num - 1,
                    final_status="error",
                    success=False,
                    images=iteration_images
                )
            logger.info(f"[ReAct] Got image: {image_path}")

            # Determine scene type for this iteration
            # Only detect scene once at the first iteration when scene_type is AUTO
            effective_scene_type = session.scene_type
            if session.scene_type == SceneType.AUTO:
                if iteration_num == 1:
                    logger.info(f"[ReAct] First iteration with AUTO scene, detecting scene...")
                    from ops_core.prompts import SceneDetector
                    detector = SceneDetector(session.api_url, session.model)
                    session.react_detected_scene = detector.detect(image_path)
                    logger.info(f"[ReAct] Detected scene: {session.react_detected_scene.value}")
                effective_scene_type = session.react_detected_scene or SceneType.GENERAL

            # Build base prompt
            base_prompt = session.react_task_description

            # Use memory to build enhanced prompt
            enhanced_prompt = context_builder.build_enhanced_prompt(base_prompt, iteration_num)

            # Get API response using cached client
            logger.info(f"[ReAct] Calling LLM API to check task completion...")
            api_client = session.get_llm_api_client(model=model)
            response = api_client.get_response(
                enhanced_prompt,
                image_path=image_path,
                retrieved_docs=session.retriever.retrieve(enhanced_prompt) if rag_enabled and session.retriever else None,
                scene_type=effective_scene_type
            )
            logger.info(f"[ReAct] LLM Response: {response[:200]}...")

            # Parse response
            parser = ResponseParser()
            action, element, input_content, key_content = parser.extract_action_and_element(response)
            logger.info(f"[ReAct] Parsed action: {action}, element: {element}, input: {input_content}, key: {key_content}")

            # Extract reasoning
            reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
            reasoning_matches = [m.strip() for m in __import__('re').findall(reasoning_pattern, response, __import__('re').DOTALL)]
            reasoning = reasoning_matches[0] if reasoning_matches else None

            # Check if task is completed
            task_status_pattern = r'<task_status>(.*?)</task_status>'
            task_status_matches = [m.strip() for m in __import__('re').findall(task_status_pattern, response, __import__('re').DOTALL)]
            task_status = task_status_matches[0] if task_status_matches else "in_progress"
            
            # Record iteration
            iteration_record = IterationRecord(
                iteration_number=iteration_num,
                timestamp=datetime.now().isoformat(),
                image_path=image_path,
                prompt=enhanced_prompt,
                llm_response=response,
                parsed_action=action,
                parsed_element=element,
                parsed_input=input_content,
                parsed_key=key_content,
                reasoning=reasoning,
                execution_success=False,
                task_status=task_status
            )
            
            if task_status == "completed":
                print(f"[ReAct Session {request.session_id}] Task completed at iteration {iteration_num}")
                session.react_is_running = False
                
                # Extract final_reasoning from response
                final_reasoning_pattern = r'<final_reasoning>(.*?)</final_reasoning>'
                final_reasoning_matches = [m.strip() for m in __import__('re').findall(final_reasoning_pattern, response, __import__('re').DOTALL)]
                final_reasoning = final_reasoning_matches[0] if final_reasoning_matches else None
                
                # Update iteration record
                iteration_record.execution_success = True
                session.react_memory_store.add_iteration(session.session_id, iteration_record)
                session.react_memory_store.finalize_memory(session.session_id, "completed", final_reasoning)
                
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

            # Process UI actions
            logger.info(f"[ReAct] Processing UI action...")
            execution_success = False
            execution_result = None
            execution_error = None

            if action in ["Click", "Double Click", "Right Click"] and element and image_path:
                logger.info(f"[ReAct] Executing {action} on {element}")
                executor = session.get_command_executor()
                success, result = executor.process_ui_element_request(
                    image_path, element, action, element
                )
                if success:
                    logger.info(f"[ReAct] UI action successful: {result}")
                    execution_success = True
                    execution_result = str(result)
                    if isinstance(result, str) and os.path.exists(result):
                        iteration_images.append(image_to_base64(result))
                        logger.info(f"[ReAct] Added processed image to iteration images")
                else:
                    logger.error(f"[ReAct] UI action failed: {result}")
                    execution_error = str(result)
            elif action == "Input" and input_content:
                logger.info(f"[ReAct] Sending input: {input_content}")
                script_command = f'Send "{input_content}"'
                image_server_client.send_script_command(script_command)
                logger.info(f"[ReAct] Input sent successfully")
                execution_success = True
                execution_result = "Input sent successfully"
            elif action == "Keyboard" and key_content:
                logger.info(f"[ReAct] Sending keyboard key: {key_content}")
                script_command = f'Send "{{{key_content}}}"'
                image_server_client.send_script_command(script_command)
                logger.info(f"[ReAct] Keyboard command sent successfully")
                execution_success = True
                execution_result = "Keyboard command sent successfully"

            # Update iteration record
            iteration_record.execution_success = execution_success
            iteration_record.execution_result = execution_result
            iteration_record.execution_error = execution_error
            session.react_memory_store.add_iteration(session.session_id, iteration_record)
            session.react_memory_store.update_patterns(session.session_id, iteration_record)

            # Wait for system to respond
            import time
            logger.info(f"[ReAct] Waiting for system to respond...")
            time.sleep(3)

        # Maximum iterations reached
        logger.warning(f"[ReAct] Maximum iterations reached: {session.react_max_iterations}")
        session.react_is_running = False
        session.react_memory_store.finalize_memory(session.session_id, "max_iterations")
        return ReactResponse(
            message="Maximum iterations reached",
            iterations_completed=session.react_max_iterations,
            final_status="max_iterations",
            success=False,
            images=iteration_images
        )

    except Exception as e:
        logger.error(f"[ReAct] Error during execution: {str(e)}", exc_info=True)
        session.react_is_running = False
        session.react_memory_store.finalize_memory(session.session_id, "error", str(e))
        return ReactResponse(
            message=f"Error during ReAct execution: {str(e)}",
            iterations_completed=session.react_current_iteration,
            final_status="error",
            success=False,
            images=iteration_images
        )

@router.post("/stop-react", response_model=ReactResponse)
async def stop_react(request: StopReactRequest):
    """Stop ReAct agent"""
    # Get session
    session = session_manager.get_session(request.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Stop ReAct
    if session.react_is_running:
        session.react_is_running = False
        if session.react_memory:
            session.react_memory_store.finalize_memory(session.session_id, "stopped")
        return ReactResponse(
            message="ReAct agent stopped",
            iterations_completed=session.react_current_iteration,
            final_status="stopped",
            success=True
        )
    else:
        return ReactResponse(
            message="ReAct agent is not running",
            iterations_completed=session.react_current_iteration,
            final_status="not_running",
            success=False
        )


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session manually"""
    success = await session_manager.delete_session(session_id)
    if success:
        return {"message": "Session deleted", "session_id": session_id, "success": True}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@router.delete("/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task manually"""
    success = await task_manager.remove_task(task_id)
    if success:
        return {"message": "Task deleted", "task_id": task_id, "success": True}
    else:
        raise HTTPException(status_code=404, detail="Task not found")
