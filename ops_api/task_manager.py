"""
ReAct task manager
Manages creation, execution, stopping, and status querying of async tasks
"""
import asyncio
import logging
import os
import re
import uuid
from typing import Dict, Optional, List
from datetime import datetime, timedelta

from ops_core import (
    ResponseParser
)
from ops_core.ui_operations.executor import CommandExecutor, CommandBuilder
from ops_core.prompts import SceneType
from ops_api.react_context import ReActContextBuilder
from ops_api.react_memory import IterationRecord
from config import Config

logger = logging.getLogger(__name__)


def image_to_base64(filepath: str) -> str:
    """Convert image file to base64 string"""
    with open(filepath, "rb") as f:
        import base64
        return base64.b64encode(f.read()).decode("utf-8")


class ReActTask:
    """ReAct task class"""

    def __init__(self, task_id: str, session_id: str, task_description: str, max_iterations: int, session):
        self.task_id = task_id
        self.session_id = session_id
        self.task_description = task_description
        self.max_iterations = max_iterations
        self.session = session

        # Status
        self.status = "pending"  # pending, running, waiting_approval, completed, stopped, error, rejected
        self.current_iteration = 0
        self.final_status = None
        self.message = None
        self.error_message = None

        # Last execution information
        self.last_action = None
        self.last_element = None
        self.last_key_content = None
        self.last_reasoning = None
        self.last_task_status = None
        self.last_image = None
        self.last_status = None

        # Approval related
        self.pending_approval = None
        self.approval_history = []
        self.approval_policy = "manual"  # auto, manual, strict
        self.dangerous_actions = [
            "Delete", "Format", "Uninstall", "Remove",
            "Erase", "Wipe", "Clear", "Reset", "Destroy",
            "删除", "格式化", "卸载", "移除",
            "擦除", "清除", "重置", "销毁"
        ]

        # Control flags
        self.should_stop = False
        self.approval_result = None  # approved, rejected

        # Timestamps
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # Iteration images
        self.iteration_images = []

        # Event queue (for SSE push)
        self.event_queue = asyncio.Queue()

        # Approval event (for synchronous waiting)
        self.approval_event = asyncio.Event()

    def update_progress(self, **kwargs):
        """Update progress"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        self.updated_at = datetime.now()

            # Push update to event queue
        try:
            self.event_queue.put_nowait({
                "type": "progress",
                "data": kwargs
            })
        except asyncio.QueueFull:
            pass

    def complete(self, final_status: str, message: str):
        """Complete task"""
        self.status = "completed"
        self.final_status = final_status
        self.message = message
        self.updated_at = datetime.now()

        try:
            self.event_queue.put_nowait({
                "type": "completed",
                "data": {
                    "final_status": final_status,
                    "message": message
                }
            })
        except asyncio.QueueFull:
            pass

    def stop(self):
        """Stop task"""
        self.status = "stopped"
        self.should_stop = True
        self.updated_at = datetime.now()

        try:
            self.event_queue.put_nowait({
                "type": "stopped",
                "data": {}
            })
        except asyncio.QueueFull:
            pass

    def error(self, error_message: str):
        """Task error"""
        self.status = "error"
        self.error_message = error_message
        self.updated_at = datetime.now()

        try:
            self.event_queue.put_nowait({
                "type": "error",
                "data": {"error": error_message}
            })
        except asyncio.QueueFull:
            pass

    def request_approval(self, approval_data):
        """Request approval"""
        self.status = "waiting_approval"
        self.pending_approval = approval_data
        self.updated_at = datetime.now()

        try:
            self.event_queue.put_nowait({
                "type": "approval_required",
                "data": approval_data
            })
        except asyncio.QueueFull:
            pass

        logger.info(f"[TaskManager] Approval requested for task {self.task_id}")

    def approve(self):
        """Approve action"""
        self.status = "running"
        self.approval_result = "approved"
        self.pending_approval = None
        self.updated_at = datetime.now()

        self.approval_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": self.last_action,
            "element": self.last_element,
            "key_content": self.last_key_content,
            "decision": "approved",
            "iteration": self.current_iteration
        })

        self.approval_event.set()
        self.approval_event.clear()

        logger.info(f"[TaskManager] Action approved for task {self.task_id}")

    def reject(self, reason: str):
        """Reject action"""
        self.status = "rejected"
        self.approval_result = "rejected"
        self.should_stop = True
        self.updated_at = datetime.now()

        self.approval_history.append({
            "timestamp": datetime.now().isoformat(),
            "action": self.last_action,
            "element": self.last_element,
            "key_content": self.last_key_content,
            "decision": "rejected",
            "reason": reason,
            "iteration": self.current_iteration
        })

        self.approval_event.set()
        self.approval_event.clear()

        logger.info(f"[TaskManager] Action rejected for task {self.task_id}: {reason}")

    async def wait_for_approval(self, timeout: int = 300) -> str:
        """Wait for approval result"""
        try:
            await asyncio.wait_for(self.approval_event.wait(), timeout=timeout)
            return self.approval_result
        except asyncio.TimeoutError:
            logger.warning(f"[TaskManager] Approval timeout for task {self.task_id}")
            self.should_stop = True
            return "timeout"


class ReActTaskManager:
    """ReAct task manager"""

    def __init__(self):
        self.tasks: Dict[str, ReActTask] = {}
        self.lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = True

    def start_cleanup_task(self):
        """Start background cleanup task"""
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            logger.info("[TaskManager] Started background cleanup task")

    def stop_cleanup_task(self):
        """Stop background cleanup task"""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            logger.info("[TaskManager] Stopped background cleanup task")

    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while self._running:
            try:
                await asyncio.sleep(Config.CLEANUP_INTERVAL_SECONDS)
                await self.cleanup_expired_tasks()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"[TaskManager] Cleanup loop error: {e}")

    async def cleanup_expired_tasks(self):
        """Clean up expired tasks"""
        now = datetime.now()
        ttl = timedelta(seconds=Config.TASK_TTL_SECONDS)
        expired_task_ids = []

        async with self.lock:
            for task_id, task in list(self.tasks.items()):
                if task.status in ["completed", "stopped", "error", "rejected"]:
                    if now - task.updated_at > ttl:
                        expired_task_ids.append(task_id)

            for task_id in expired_task_ids:
                del self.tasks[task_id]
                logger.info(f"[TaskManager] Cleaned up expired task {task_id}")

        if expired_task_ids:
            logger.info(f"[TaskManager] Cleaned up {len(expired_task_ids)} expired tasks")

    async def remove_task(self, task_id: str) -> bool:
        """Remove a task manually"""
        async with self.lock:
            if task_id in self.tasks:
                del self.tasks[task_id]
                logger.info(f"[TaskManager] Manually removed task {task_id}")
                return True
        return False

    async def create_task(self, session_id: str, task_description: str, max_iterations: int, session, approval_policy: str = "manual") -> str:
        """Create new task"""
        async with self.lock:
            if len(self.tasks) >= Config.MAX_TASKS:
                self._evict_oldest_completed_task()

            task_id = str(uuid.uuid4())
            task = ReActTask(
                task_id=task_id,
                session_id=session_id,
                task_description=task_description,
                max_iterations=max_iterations,
                session=session
            )
            task.approval_policy = approval_policy

            self.tasks[task_id] = task

        logger.info(f"[TaskManager] Created task {task_id} for session {session_id} with policy {approval_policy}")
        return task_id

    def _evict_oldest_completed_task(self):
        """Evict the oldest completed task when limit is reached (must be called with lock held)"""
        completed_tasks = [
            (task_id, task.updated_at)
            for task_id, task in self.tasks.items()
            if task.status in ["completed", "stopped", "error", "rejected"]
        ]

        if completed_tasks:
            completed_tasks.sort(key=lambda x: x[1])
            oldest_task_id = completed_tasks[0][0]
            del self.tasks[oldest_task_id]
            logger.info(f"[TaskManager] Evicted oldest completed task {oldest_task_id} to make room")

    async def start_task(self, task_id: str):
        """Start task"""
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"[TaskManager] Task {task_id} not found")
            return

        task.status = "running"
        # Clear cached detected scene for new task
        task.session.react_detected_scene = None
        logger.info(f"[TaskManager] Starting task {task_id}")

        # Execute task in background
        asyncio.create_task(self._execute_task(task))

    async def stop_task(self, task_id: str):
        """Stop task"""
        task = self.tasks.get(task_id)
        if task:
            task.stop()
            logger.info(f"[TaskManager] Stopping task {task_id}")

    async def approve_action(self, task_id: str):
        """Approve action"""
        task = self.tasks.get(task_id)
        if task and task.status == "waiting_approval":
            task.approve()
            return True
        return False

    async def reject_action(self, task_id: str, reason: str):
        """Reject action"""
        task = self.tasks.get(task_id)
        if task and task.status == "waiting_approval":
            task.reject(reason)
            return True
        return False

    def set_approval_policy(self, task_id: str, policy: str):
        """Set approval policy"""
        task = self.tasks.get(task_id)
        if task:
            task.approval_policy = policy
            logger.info(f"[TaskManager] Set approval policy to {policy} for task {task_id}")

            if policy == "auto" and task.status == "waiting_approval":
                logger.info(f"[TaskManager] Auto-approving current pending action for task {task_id}")
                task.approve()

    def get_task(self, task_id: str) -> Optional[ReActTask]:
        """Get task"""
        return self.tasks.get(task_id)

    def _is_dangerous_action(self, action: str, element: str, key_content: str, dangerous_keywords: List[str]) -> bool:
        """Check if dangerous action"""
        dangerous_action_types = [
            "delete", "format", "uninstall", "destroy",
            "删除", "格式化", "卸载", "销毁"
        ]
        if action and action.lower() in dangerous_action_types:
            return True

        if element:
            element_lower = element.lower()
            if any(keyword.lower() in element_lower for keyword in dangerous_keywords):
                return True
        
        if key_content:
            key_content_lower = key_content.lower()
            if any(keyword.lower() in key_content_lower for keyword in dangerous_keywords):
                return True

        return False

    async def _execute_task(self, task: ReActTask):
        """Execute task (async)"""
        try:
            logger.info(f"[TaskManager] Executing task {task.task_id}")

            # Initialize ReAct memory
            task.session.initialize_react_memory(task.task_description)
            context_builder = ReActContextBuilder(
                task.session.react_memory,
                max_context_iterations=10
            )

            # Use session model and RAG settings
            model = task.session.model
            rag_enabled = task.session.rag_enabled

            logger.info(f"[TaskManager] Model: {model}, RAG enabled: {rag_enabled}")

            # Get cached client instances from session
            image_server_client = task.session.image_server_client

            # Execute iteration loop
            while not task.should_stop and task.current_iteration < task.max_iterations:
                task.current_iteration += 1
                iteration_num = task.current_iteration

                logger.info(f"[TaskManager] Iteration {iteration_num}/{task.max_iterations}")

                # Fetch latest image from server using cached client
                # 使用 gettargetscreen 命令获取屏幕图像
                image_path = image_server_client.get_target_screen()

                # 错误检查
                if not image_path or image_path.startswith("Error:") or not image_path.startswith("./images"):
                    logger.error(f"[TaskManager] Failed to get image: {image_path}")
                    task.session.react_memory_store.finalize_memory(task.session_id, "error")
                    task.error(f"Failed to get image from server: {image_path}")
                    return

                logger.info(f"[TaskManager] Got image: {image_path}")

                # Determine scene type for this iteration
                # Only detect scene once at the first iteration when scene_type is AUTO
                effective_scene_type = task.session.scene_type
                logger.info(f"[TaskManager] Iteration {iteration_num}, session.scene_type = {task.session.scene_type} (value: {task.session.scene_type.value if hasattr(task.session.scene_type, 'value') else task.session.scene_type})")
                
                if task.session.scene_type == SceneType.AUTO:
                    if iteration_num == 1:
                        logger.info(f"[TaskManager] First iteration with AUTO scene, detecting scene...")
                        from ops_core.prompts import SceneDetector
                        detector = SceneDetector(task.session.api_url, task.session.model)
                        task.session.react_detected_scene = detector.detect(image_path)
                        logger.info(f"[TaskManager] Detected scene: {task.session.react_detected_scene.value}")
                    effective_scene_type = task.session.react_detected_scene or SceneType.GENERAL
                    logger.info(f"[TaskManager] Using effective_scene_type (detected): {effective_scene_type.value}")
                else:
                    logger.info(f"[TaskManager] Using user-selected scene_type: {effective_scene_type.value}")

                # Build prompt
                base_prompt = task.task_description

                # Use memory to build enhanced prompt
                enhanced_prompt = context_builder.build_enhanced_prompt(
                    base_prompt,
                    iteration_num
                )

                # Get API response using cached client
                api_client = task.session.get_llm_api_client(model=model)
                response = api_client.get_response(
                    enhanced_prompt,
                    image_path=image_path,
                    retrieved_docs=task.session.retriever.retrieve(enhanced_prompt)
                    if rag_enabled and task.session.retriever else None,
                    scene_type=effective_scene_type
                )

                logger.info(f"[TaskManager] LLM Response: {response[:200]}...")

                # Parse response
                parser = ResponseParser()
                action, element, input_content, key_content, point_coords = parser.extract_action_and_element(response)

                logger.info(f"[TaskManager] Parsed: action={action}, element={element}, point={point_coords}")

                # Extract reasoning
                reasoning_pattern = r'<reasoning>(.*?)</reasoning>'
                reasoning_matches = [m.strip() for m in re.findall(
                    reasoning_pattern, response, re.DOTALL
                )]
                reasoning = reasoning_matches[0] if reasoning_matches else None

                # Check task status
                task_status_pattern = r'<task_status>(.*?)</task_status>'
                task_status_matches = [m.strip() for m in re.findall(
                    task_status_pattern, response, re.DOTALL
                )]
                task_status = task_status_matches[0] if task_status_matches else "in_progress"

                # Extract task plan (only on first iteration or if no plan exists yet)
                plan_data = None
                if iteration_num == 1 or not task.session.react_memory.task_plan:
                    plan_data = parser.parse_plan(response)
                    if plan_data:
                        from ops_api.react_memory import TaskPlan, SubTask
                        plan = TaskPlan(
                            overview=plan_data["overview"],
                            subtasks=[
                                SubTask(id=str(st["id"]), description=st["description"])
                                for st in plan_data["subtasks"]
                            ]
                        )
                        task.session.react_memory_store.set_task_plan(task.session_id, plan)
                        logger.info(f"[TaskManager] Task plan created: {plan.overview}, {len(plan.subtasks)} subtasks")

                # Extract subtask status update
                status_update = parser.parse_subtask_status_update(response)
                if status_update:
                    updated = task.session.react_memory_store.update_subtask_status(
                        task.session_id, status_update["id"], status_update["status"],
                        notes=status_update.get("notes", ""), iteration=iteration_num
                    )
                    if updated:
                        logger.info(f"[TaskManager] Subtask {status_update['id']} status updated to {status_update['status']}, notes: {status_update.get('notes', '')}")

                # Extract dynamic plan update (add/remove/modify subtask, update overview)
                plan_update = parser.parse_plan_update(response)
                if plan_update:
                    applied = task.session.react_memory_store.apply_plan_update(task.session_id, plan_update)
                    if applied:
                        op = plan_update["operation"]
                        if op == "add":
                            logger.info(f"[TaskManager] Plan updated: added subtask '{plan_update['description']}'")
                        elif op == "remove":
                            logger.info(f"[TaskManager] Plan updated: removed subtask {plan_update['subtask_id']}")
                        elif op == "modify":
                            logger.info(f"[TaskManager] Plan updated: modified subtask {plan_update['subtask_id']}")
                        elif op == "update_overview":
                            logger.info(f"[TaskManager] Plan updated: overview changed to '{plan_update['overview']}'")

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

                # Check if need to stop
                if task.should_stop:
                    logger.info(f"[TaskManager] Task stop requested before executing action")
                    task.session.react_memory_store.finalize_memory(task.session_id, "stopped")
                    task.stop()
                    return

                # Check if task is completed
                if task_status == "completed":
                    logger.info(f"[TaskManager] Task completed at iteration {iteration_num}")

                    # Extract final_reasoning
                    final_reasoning_pattern = r'<final_reasoning>(.*?)</final_reasoning>'
                    final_reasoning_matches = [m.strip() for m in re.findall(
                        final_reasoning_pattern, response, re.DOTALL
                    )]
                    final_reasoning = final_reasoning_matches[0] if final_reasoning_matches else None

                    # Auto-complete any remaining subtasks
                    plan = task.session.react_memory.task_plan
                    if plan:
                        for st in plan.subtasks:
                            if st.status != "completed":
                                plan.update_subtask_status(st.id, "completed", iteration=iteration_num)
                        logger.info(f"[TaskManager] Task plan finalized: {plan.completed_count}/{plan.total_count} subtasks completed")

                    # Update iteration record
                    iteration_record.execution_success = True
                    task.session.react_memory_store.add_iteration(task.session_id, iteration_record)
                    task.session.react_memory_store.finalize_memory(
                        task.session_id,
                        "completed",
                        final_reasoning
                    )

                    # Complete task
                    message = f"Task completed successfully in {iteration_num} iterations"
                    if final_reasoning:
                        message += f"\n<final_reasoning>{final_reasoning}</final_reasoning>"

                    task.complete("completed", message)
                    return

                # Save last execution information
                task.last_action = action
                task.last_element = element
                task.last_key_content = key_content
                task.last_reasoning = reasoning
                task.last_task_status = task_status
                task.last_image = image_path

                # Danger action detection and approval
                needs_approval = False
                is_dangerous = False

                if task.approval_policy == "strict":
                    needs_approval = True
                    is_dangerous = True
                elif task.approval_policy == "manual":
                    is_dangerous = self._is_dangerous_action(action, element, key_content, task.dangerous_actions)
                    needs_approval = is_dangerous

                if needs_approval and action and action != "none":
                    logger.info(f"[TaskManager] Action requires approval: {action} on {element}")

                    # Request approval
                    approval_data = {
                        "iteration": iteration_num,
                        "action": action,
                        "element": element,
                        "key_content": key_content,
                        "reasoning": reasoning,
                        "is_dangerous": is_dangerous,
                        "image": image_to_base64(image_path) if image_path else None
                    }
                    task.request_approval(approval_data)

                    # Wait for approval result
                    approval_result = await task.wait_for_approval(timeout=300)

                    if approval_result == "rejected":
                        logger.info(f"[TaskManager] Action rejected, stopping task")
                        task.session.react_memory_store.finalize_memory(task.session_id, "rejected")
                        task.error(f"Action rejected by user: {task.approval_history[-1].get('reason', 'No reason')}")
                        return
                    elif approval_result == "timeout":
                        logger.info(f"[TaskManager] Approval timeout, stopping task")
                        task.session.react_memory_store.finalize_memory(task.session_id, "timeout")
                        task.error("Approval timeout")
                        return
                    elif approval_result == "approved":
                        logger.info(f"[TaskManager] Action approved, continuing execution")

                # Process UI actions
                execution_success = False
                execution_result = None
                execution_error = None
                current_image_base64 = None  # Current operation image for real-time display

                # Check if this is a Sequence operation
                if action == "Sequence":
                    logger.info(f"[TaskManager] Executing sequence operation")
                    executor = task.session.get_command_executor()
                    parser = ResponseParser()

                    # Parse sequence operations from response
                    sequence_ops = parser.parse_sequence_operations(response)

                    if sequence_ops:
                        logger.info(f"[TaskManager] Parsed {len(sequence_ops)} steps from sequence")

                        # Build and execute command sequence (no UI-Ins client needed)
                        success, result = executor.execute_sequence_operations(
                            sequence_ops=sequence_ops,
                            image_path=image_path
                        )

                        if success:
                            logger.info(f"[TaskManager] Sequence execution successful: {result}")
                            execution_success = True
                            execution_result = str(result)
                            if isinstance(result, str) and os.path.exists(result):
                                task.iteration_images.append(image_to_base64(result))
                                current_image_base64 = image_to_base64(result)
                        else:
                            logger.error(f"[TaskManager] Sequence execution failed: {result}")
                            execution_error = str(result)
                    else:
                        logger.error(f"[TaskManager] No operations found in sequence")
                        execution_error = "Failed to parse sequence operations"

                elif action in ["Click", "Double Click", "Right Click"] and element and image_path:
                    logger.info(f"[TaskManager] Executing {action} on {element} at point {point_coords}")
                    executor = task.session.get_command_executor()

                    # Use point coordinates from LLM if available
                    if point_coords:
                        success, result = executor.execute_click_at_point(
                            image_path, action, point_coords
                        )
                    else:
                        # Fallback: use element description (should not happen with new model)
                        logger.warning(f"[TaskManager] No point coordinates provided, using element description fallback")
                        success, result = executor.process_ui_element_request(
                            image_path, element, action, element
                        )

                    if success:
                        logger.info(f"[TaskManager] UI action successful: {result}")
                        execution_success = True
                        execution_result = str(result)
                        if isinstance(result, str) and os.path.exists(result):
                            current_image_base64 = image_to_base64(result)
                            task.iteration_images.append(current_image_base64)
                    else:
                        logger.error(f"[TaskManager] UI action failed: {result}")
                        execution_error = str(result)
                elif action == "Input" and input_content:
                    logger.info(f"[TaskManager] Sending input: {input_content} at point {point_coords}")
                    executor = task.session.get_command_executor()

                    # If point coordinates are provided, click first then type
                    if point_coords:
                        success, result = executor.execute_input_at_point(
                            image_path, point_coords, input_content
                        )
                        if success:
                            logger.info(f"[TaskManager] Input action successful: {result}")
                            execution_success = True
                            execution_result = str(result)
                            if isinstance(result, str) and os.path.exists(result):
                                current_image_base64 = image_to_base64(result)
                                task.iteration_images.append(current_image_base64)
                        else:
                            logger.error(f"[TaskManager] Input action failed: {result}")
                            execution_error = str(result)
                    else:
                        # Fallback: just send text without clicking
                        script_command = f'Send "{input_content}"'
                        image_server_client.send_script_command(script_command)
                        logger.info(f"[TaskManager] Input sent successfully (fallback mode)")
                        execution_success = True
                        execution_result = "Input sent successfully"
                elif action == "Move Mouse" and point_coords:
                    logger.info(f"[TaskManager] Moving mouse to point {point_coords}")
                    executor = task.session.get_command_executor()
                    success, result = executor.execute_move_mouse(image_path, point_coords)
                    if success:
                        logger.info(f"[TaskManager] Move Mouse action successful: {result}")
                        execution_success = True
                        execution_result = str(result)
                    else:
                        logger.error(f"[TaskManager] Move Mouse action failed: {result}")
                        execution_error = str(result)
                elif action == "Keyboard" and key_content:
                    logger.info(f"[TaskManager] Sending keyboard key: {key_content}")
                    from ops_core.utils.key_map import is_combo_key, get_tcp_key_code
                    key_code = get_tcp_key_code(key_content)
                    if is_combo_key(key_content):
                        script_command = f'Send {key_code}'
                    else:
                        script_command = f'Send "{key_code}"'
                    image_server_client.send_script_command(script_command)
                    logger.info(f"[TaskManager] Keyboard command sent: {script_command}")
                    execution_success = True
                    execution_result = "Keyboard command sent successfully"

                    # Capture screenshot after keyboard operation
                    try:
                        executor = task.session.get_command_executor()
                        screenshot_path = executor.capture_screenshot("keyboard_after")
                        if screenshot_path and os.path.exists(screenshot_path):
                            current_image_base64 = image_to_base64(screenshot_path)
                            task.iteration_images.append(current_image_base64)
                            logger.info(f"[TaskManager] Keyboard screenshot captured: {screenshot_path}")
                    except Exception as e:
                        logger.error(f"[TaskManager] Keyboard screenshot capture failed: {str(e)}", exc_info=True)
                elif action == "Scroll":
                    direction = key_content if key_content else "down"
                    direction = direction.lower()
                    if direction not in ("up", "down"):
                        direction = "down"
                    logger.info(f"[TaskManager] Sending scroll command: direction={direction}")
                    executor = task.session.get_command_executor()
                    success, result = executor.scroll(direction)
                    if success:
                        logger.info(f"[TaskManager] Scroll action successful: {result}")
                        execution_success = True
                        execution_result = str(result)
                    else:
                        logger.error(f"[TaskManager] Scroll action failed: {result}")
                        execution_error = str(result)

                    # Capture screenshot after scroll operation
                    try:
                        screenshot_path = executor.capture_screenshot("scroll_after")
                        if screenshot_path and os.path.exists(screenshot_path):
                            current_image_base64 = image_to_base64(screenshot_path)
                            task.iteration_images.append(current_image_base64)
                            logger.info(f"[TaskManager] Scroll screenshot captured: {screenshot_path}")
                    except Exception as e:
                        logger.error(f"[TaskManager] Scroll screenshot capture failed: {str(e)}", exc_info=True)
                elif action == "Type" and input_content:
                    # Handle Type action (new action type for text input)
                    logger.info(f"[TaskManager] Sending Type action: {input_content}")
                    executor = task.session.get_command_executor()
                    success, result = executor.type_text(input_content)
                    if success:
                        logger.info(f"[TaskManager] Type action successful: {result}")
                        execution_success = True
                        execution_result = str(result)

                        # Capture screenshot after type operation
                        try:
                            screenshot_path = executor.capture_screenshot("type_after")
                            if screenshot_path and os.path.exists(screenshot_path):
                                current_image_base64 = image_to_base64(screenshot_path)
                                task.iteration_images.append(current_image_base64)
                                logger.info(f"[TaskManager] Type screenshot captured: {screenshot_path}")
                        except Exception as e:
                            logger.error(f"[TaskManager] Type screenshot capture failed: {str(e)}", exc_info=True)
                    else:
                        logger.error(f"[TaskManager] Type action failed: {result}")
                        execution_error = str(result)
                elif action == "Press" and key_content:
                    # Handle Press action (new action type for keypress)
                    logger.info(f"[TaskManager] Sending Press action: {key_content}")
                    executor = task.session.get_command_executor()
                    success, result = executor.press_key(key_content)
                    if success:
                        logger.info(f"[TaskManager] Press action successful: {result}")
                        execution_success = True
                        execution_result = str(result)

                        # Capture screenshot after press operation
                        try:
                            screenshot_path = executor.capture_screenshot("press_after")
                            if screenshot_path and os.path.exists(screenshot_path):
                                current_image_base64 = image_to_base64(screenshot_path)
                                task.iteration_images.append(current_image_base64)
                                logger.info(f"[TaskManager] Press screenshot captured: {screenshot_path}")
                        except Exception as e:
                            logger.error(f"[TaskManager] Press screenshot capture failed: {str(e)}", exc_info=True)
                    else:
                        logger.error(f"[TaskManager] Press action failed: {result}")
                        execution_error = str(result)
                elif action == "Wait":
                    logger.info(f"[TaskManager] Waiting as instructed by reasoning...")
                    execution_success = True
                    execution_result = "Wait completed"

                    # Capture screenshot during wait to show current state
                    try:
                        executor = task.session.get_command_executor()
                        screenshot_path = executor.capture_screenshot("wait_state")
                        if screenshot_path and os.path.exists(screenshot_path):
                            current_image_base64 = image_to_base64(screenshot_path)
                            task.iteration_images.append(current_image_base64)
                            logger.info(f"[TaskManager] Wait state screenshot captured: {screenshot_path}")
                    except Exception as e:
                        logger.error(f"[TaskManager] Wait state screenshot capture failed: {str(e)}", exc_info=True)

                # Update iteration record
                iteration_record.execution_success = execution_success
                iteration_record.execution_result = execution_result
                iteration_record.execution_error = execution_error
                task.session.react_memory_store.add_iteration(task.session_id, iteration_record)
                task.session.react_memory_store.update_patterns(task.session_id, iteration_record)

                # Update progress: display operation result
                # Serialize task plan for frontend
                plan_for_frontend = None
                if task.session.react_memory.task_plan:
                    plan = task.session.react_memory.task_plan
                    plan_for_frontend = {
                        "overview": plan.overview,
                        "subtasks": [
                            {"id": st.id, "description": st.description, "status": st.status, "notes": st.notes}
                            for st in plan.subtasks
                        ],
                        "completed_count": plan.completed_count,
                        "total_count": plan.total_count
                    }

                # Serialize subtask status update for frontend
                subtask_update_for_frontend = None
                if status_update:
                    subtask_update_for_frontend = {
                        "id": status_update["id"],
                        "status": status_update["status"],
                        "notes": status_update.get("notes", "")
                    }

                task.update_progress(
                    iteration=iteration_num,
                    action=action,
                    element=element,
                    key_content=key_content,
                    reasoning=reasoning,
                    task_status=task_status,
                    image=current_image_base64,
                    task_plan=plan_for_frontend,
                    subtask_status_update=subtask_update_for_frontend
                )

                # Check if need to stop
                if task.should_stop:
                    logger.info(f"[TaskManager] Task stop requested before waiting")
                    task.session.react_memory_store.finalize_memory(task.session_id, "stopped")
                    task.stop()
                    return

                # Wait for system to respond (dynamic based on action type)
                wait_seconds = 3 if action == "Wait" else 1
                logger.info(f"[TaskManager] Waiting {wait_seconds}s for system to respond (action: {action})...")

                # Wait in segments, check stop flag each second
                for _ in range(wait_seconds):
                    if task.should_stop:
                        logger.info(f"[TaskManager] Task stop requested during wait")
                        task.session.react_memory_store.finalize_memory(task.session_id, "stopped")
                        task.stop()
                        return
                    await asyncio.sleep(1)

            # Maximum iterations reached
            logger.warning(f"[TaskManager] Maximum iterations reached: {task.max_iterations}")
            task.session.react_memory_store.finalize_memory(task.session_id, "max_iterations")
            task.complete("max_iterations", "Maximum iterations reached")

        except Exception as e:
            logger.error(f"[TaskManager] Error during execution: {str(e)}", exc_info=True)
            task.session.react_memory_store.finalize_memory(task.session_id, "error", str(e))
            task.error(f"Error during ReAct execution: {str(e)}")


# Global task manager instance
task_manager = ReActTaskManager()
