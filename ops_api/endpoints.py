"""
API端点实现
"""
import os
import base64
import uuid
import logging
import asyncio
from datetime import datetime
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
    LLMAPIClient,
    ImageServerClient,
    ResponseParser,
    CommandExecutor
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

# 全局会话字典
sessions: Dict[str, Session] = {}

# 创建FastAPI应用
from fastapi import APIRouter
router = APIRouter()

# 辅助函数
def save_base64_image(base64_str: str) -> str:
    """保存base64编码的图像到文件"""
    # 创建images目录
    os.makedirs(Config.IMAGES_DIR, exist_ok=True)

    # 生成文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"api_image_{timestamp}_{uuid.uuid4().hex[:8]}.jpg"
    filepath = os.path.join(Config.IMAGES_DIR, filename)

    # 解码并保存
    with open(filepath, "wb") as f:
        f.write(base64.b64decode(base64_str))

    return filepath

def image_to_base64(filepath: str) -> str:
    """将图像文件转换为base64字符串"""
    with open(filepath, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# API端点
@router.post("/create-session", response_model=SessionResponse)
async def create_session(request: CreateSessionRequest):
    """创建新会话"""
    session_id = str(uuid.uuid4())
    session = Session(session_id)

    # 应用自定义设置
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

    sessions[session_id] = session
    return SessionResponse(
        session_id=session_id,
        message="Session created successfully",
        success=True
    )

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """处理聊天请求"""
    logger.info(f"[Chat Request] Session: {request.session_id}, Prompt: {request.prompt[:100]}...")
    
    # 获取会话
    if request.session_id not in sessions:
        logger.error(f"[Chat] Session not found: {request.session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]

    # 使用会话或请求的模型
    model = request.model or session.model
    rag_enabled = request.rag_enabled if request.rag_enabled is not None else session.rag_enabled

    logger.info(f"[Chat] Model: {model}, RAG enabled: {rag_enabled}, Multiturn mode: {session.is_multiturn_mode}")

    # 获取图像路径
    image_path = None

    # 首先检查会话中是否有保存的图像
    if session.current_image_path:
        image_path = session.current_image_path
        logger.info(f"[Chat] Using session image: {image_path}")
    # 然后检查是否请求从服务器获取图像
    elif request.get_image_from_server:
        logger.info("[Chat] Fetching image from server...")
        image_server_client = ImageServerClient()
        image_path = image_server_client.get_last_image()
        if image_path and not image_path.startswith("./images"):
            logger.error(f"[Chat] Failed to get image from server: {image_path}")
            return ChatResponse(
                response=f"Failed to get image from server: {image_path}",
                history=session.conversation_history,
                success=False
            )
        logger.info(f"[Chat] Got image from server: {image_path}")
    # 最后检查请求中是否提供了图像
    elif request.image:
        # 保存base64图像到文件
        logger.info("[Chat] Saving base64 image to file...")
        image_path = save_base64_image(request.image)
        logger.info(f"[Chat] Saved image to: {image_path}")

    # 如果启用RAG，检索相关文档
    retrieved_docs = []
    if rag_enabled and session.retriever:
        logger.info("[Chat] Retrieving relevant documents...")
        retrieved_docs = session.retriever.retrieve(request.prompt)
        logger.info(f"[Chat] Retrieved {len(retrieved_docs)} documents")

    # 获取API响应
    logger.info(f"[Chat] Calling LLM API at {session.api_url}...")
    api_client = LLMAPIClient(session.api_url, model)
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

    # 更新对话历史（如果是多轮模式）
    updated_history = session.conversation_history.copy()
    if session.is_multiturn_mode and response:
        # 添加用户消息
        user_msg = {"role": "user", "content": request.prompt}
        if image_path:
            user_msg["content"] = [
                {"type": "text", "text": request.prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(image_path)}"}}
            ]
        updated_history.append(user_msg)
        # 添加AI响应
        updated_history.append({"role": "assistant", "content": response})
        session.conversation_history = updated_history

    # 检查UI-Model请求
    processed_image = None
    parser = ResponseParser()
    action, element, input_content, key_content = parser.extract_action_and_element(response)
    
    logger.info(f"[Chat] Parsed action: {action}, element: {element}, input: {input_content}, key: {key_content}")

    if action in ["Click", "Double Click", "Right Click"] and element and image_path:
        # 处理UI-Model
        logger.info(f"[Chat] Processing UI-Model request: {action} on {element}")
        try:
            executor = CommandExecutor(session.ui_model_api_url, session.ui_model)
            logger.info(f"[Chat] Calling UI-Model API at {session.ui_model_api_url} with model {session.ui_model}")
            success, result = executor.process_ui_element_request(
                image_path, element, action, element
            )

            if success:
                logger.info(f"[Chat] UI-Model success: {result}")
                if isinstance(result, str) and os.path.exists(result):
                    # 转换为base64
                    processed_image = image_to_base64(result)
                    logger.info(f"[Chat] Processed image generated: {result}")
            else:
                logger.error(f"[Chat] UI-Model failed: {result}")

        except Exception as e:
            logger.error(f"[Chat] UI-Model processing error: {str(e)}", exc_info=True)
    elif action == "Input" and input_content:
        # 处理Input动作 - 直接发送文本
        logger.info(f"[Chat] Sending input: {input_content}")
        image_server_client = ImageServerClient()
        script_command = f'Send "{input_content}"'
        image_server_client.send_script_command(script_command)
        logger.info(f"[Chat] Input sent successfully")
    elif action == "Keyboard" and key_content:
        # 处理Keyboard动作 - 发送按键命令
        logger.info(f"[Chat] Sending keyboard key: {key_content}")
        image_server_client = ImageServerClient()
        script_command = f'Send "{{{key_content}}}"'
        image_server_client.send_script_command(script_command)
        logger.info(f"[Chat] Keyboard command sent successfully")

    # 处理完成后清除当前图像路径
    session.current_image_path = None

    return ChatResponse(
        response=response,
        image=processed_image,
        history=updated_history,
        success=True
    )

@router.post("/build-index", response_model=IndexResponse)
async def build_index(request: BuildIndexRequest):
    """从文档构建RAG索引"""
    # 获取会话
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]

    # 构建索引
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

# 新的异步任务端点
@router.post("/react-task", response_model=CreateReactTaskResponse)
async def create_react_task(request: CreateReactTaskRequest):
    """创建并启动异步ReAct任务"""
    logger.info(f"[ReAct Task] Creating task for session {request.session_id}")
    logger.info(f"[ReAct Task] Task: {request.task}, Max iterations: {request.max_iterations}, Approval policy: {request.approval_policy}")

    # 获取会话
    if request.session_id not in sessions:
        logger.error(f"[ReAct Task] Session not found: {request.session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]

    # 应用自定义设置
    if request.model:
        session.model = request.model
    if request.rag_enabled is not None:
        session.rag_enabled = request.rag_enabled

    # 创建任务
    task_id = await task_manager.create_task(
        session_id=request.session_id,
        task_description=request.task,
        max_iterations=request.max_iterations,
        session=session,
        approval_policy=request.approval_policy
    )

    # 启动任务
    await task_manager.start_task(task_id)

    logger.info(f"[ReAct Task] Task {task_id} created and started")

    return CreateReactTaskResponse(
        task_id=task_id,
        message="ReAct task created and started",
        success=True
    )


@router.get("/react-stream/{task_id}")
async def react_stream(task_id: str):
    """SSE流式推送任务进度"""
    from fastapi.responses import StreamingResponse
    import json

    task = task_manager.get_task(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        try:
            while True:
                # 检查任务状态
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

                # 从事件队列获取更新
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
                    # 超时继续循环
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
    """获取任务状态"""
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
    """停止异步ReAct任务"""
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
    """批准操作"""
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
    """拒绝操作"""
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
    """设置审核策略"""
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
    """获取审核历史"""
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
    """获取API状态"""
    # 获取会话
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[session_id]

    # 检查API连接
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
    """切换语言"""
    # 获取会话
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]

    # 切换语言
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

@router.post("/toggle-rag", response_model=ToggleResponse)
async def toggle_rag(request: ToggleRequest):
    """切换RAG功能"""
    # 获取会话
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]

    # 切换RAG
    session.rag_enabled = not session.rag_enabled
    return ToggleResponse(
        message="RAG functionality toggled",
        current_state=session.rag_enabled,
        success=True
    )

@router.post("/toggle-multiturn", response_model=ToggleResponse)
async def toggle_multiturn(request: ToggleRequest):
    """切换多轮对话模式"""
    # 获取会话
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]

    # 如果提供了模式则设置，否则切换
    if request.mode is not None:
        session.is_multiturn_mode = request.mode
    else:
        session.is_multiturn_mode = not session.is_multiturn_mode

    # 如果切换到单轮模式，清除历史
    if not session.is_multiturn_mode:
        session.clear_history()

    return ToggleResponse(
        message="Multiturn mode toggled",
        current_state=session.is_multiturn_mode,
        success=True
    )

@router.post("/clear-history", response_model=SessionResponse)
async def clear_history(request: ClearHistoryRequest):
    """清除对话历史"""
    # 获取会话
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]

    # 清除历史
    session.clear_history()
    return SessionResponse(
        session_id=request.session_id,
        message="Conversation history cleared",
        success=True
    )

@router.post("/get-image", response_model=GetImageResponse)
async def get_image(request: GetImageRequest):
    """从服务器获取最新图像"""
    # 获取会话
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]

    # 从服务器获取图像
    image_server_client = ImageServerClient()
    image_path = image_server_client.get_last_image()
    if image_path and not image_path.startswith("./images"):
        return GetImageResponse(
            image="",
            message=f"Failed to get image from server: {image_path}",
            success=False
        )

    # 保存图像路径到会话
    session.current_image_path = image_path

    # 转换图像为base64
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
    """启动ReAct agent模式自主完成任务"""
    logger.info(f"[ReAct] Starting ReAct agent for session {request.session_id}")
    logger.info(f"[ReAct] Task: {request.task}, Max iterations: {request.max_iterations}")
    
    # 获取会话
    if request.session_id not in sessions:
        logger.error(f"[ReAct] Session not found: {request.session_id}")
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]

    # 检查是否已在运行
    if session.react_is_running:
        logger.warning(f"[ReAct] Agent already running for session {request.session_id}")
        return ReactResponse(
            message="ReAct agent is already running",
            iterations_completed=0,
            final_status="already_running",
            success=False
        )

    # 初始化ReAct模式
    session.react_enabled = True
    session.react_max_iterations = request.max_iterations
    session.react_current_iteration = 0
    session.react_task_description = request.task
    session.react_is_running = True

    # 初始化记忆
    session.initialize_react_memory(request.task)
    context_builder = ReActContextBuilder(session.react_memory, max_context_iterations=5)

    # 使用会话或请求的模型和RAG设置
    model = request.model or session.model
    rag_enabled = request.rag_enabled if request.rag_enabled is not None else session.rag_enabled

    logger.info(f"[ReAct] Model: {model}, RAG enabled: {rag_enabled}")

    # 存储每次迭代的图像
    iteration_images = []

    try:
        while session.react_is_running and session.react_current_iteration < session.react_max_iterations:
            session.react_current_iteration += 1
            iteration_num = session.react_current_iteration

            logger.info(f"[ReAct Session {request.session_id}] Iteration {iteration_num}/{session.react_max_iterations}")

            # 从服务器获取最新图像
            logger.info(f"[ReAct] Fetching image from server for iteration {iteration_num}")
            image_server_client = ImageServerClient()
            image_path = image_server_client.get_last_image()
            if image_path and not image_path.startswith("./images"):
                logger.error(f"[ReAct] Failed to get image from server: {image_path}")
                
                # 记录失败的迭代
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

            # 构建基础提示
            base_prompt = session.react_task_description

            # 使用记忆构建增强提示
            enhanced_prompt = context_builder.build_enhanced_prompt(base_prompt, iteration_num)

            # 获取API响应
            logger.info(f"[ReAct] Calling LLM API to check task completion...")
            api_client = LLMAPIClient(session.api_url, model)
            response = api_client.get_response(
                enhanced_prompt,
                image_path=image_path,
                retrieved_docs=session.retriever.retrieve(enhanced_prompt) if rag_enabled and session.retriever else None
            )
            logger.info(f"[ReAct] LLM Response: {response[:200]}...")

            # 解析响应
            parser = ResponseParser()
            action, element, input_content, key_content = parser.extract_action_and_element(response)
            logger.info(f"[ReAct] Parsed action: {action}, element: {element}, input: {input_content}, key: {key_content}")

            # 提取 reasoning
            reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
            reasoning_matches = [m.strip() for m in __import__('re').findall(reasoning_pattern, response, __import__('re').DOTALL)]
            reasoning = reasoning_matches[0] if reasoning_matches else None

            # Check if task is completed
            task_status_pattern = r'<task_status>(.*?)</task_status>'
            task_status_matches = [m.strip() for m in __import__('re').findall(task_status_pattern, response, __import__('re').DOTALL)]
            task_status = task_status_matches[0] if task_status_matches else "in_progress"
            
            # 记录迭代
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
                
                # 更新迭代记录
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

            # 处理UI操作
            logger.info(f"[ReAct] Processing UI action...")
            execution_success = False
            execution_result = None
            execution_error = None

            if action in ["Click", "Double Click", "Right Click"] and element and image_path:
                logger.info(f"[ReAct] Executing {action} on {element}")
                executor = CommandExecutor(session.ui_model_api_url, session.ui_model)
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

            # 更新迭代记录
            iteration_record.execution_success = execution_success
            iteration_record.execution_result = execution_result
            iteration_record.execution_error = execution_error
            session.react_memory_store.add_iteration(session.session_id, iteration_record)
            session.react_memory_store.update_patterns(session.session_id, iteration_record)

            # 等待一段时间让系统响应
            import time
            logger.info(f"[ReAct] Waiting for system to respond...")
            time.sleep(3)

        # 达到最大迭代次数
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
    """停止ReAct agent"""
    # 获取会话
    if request.session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    session = sessions[request.session_id]

    # 停止ReAct
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
