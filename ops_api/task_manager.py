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
from datetime import datetime

from ops_core import (
    ImageServerClient,
    LLMAPIClient,
    ResponseParser,
    CommandExecutor
)
from ops_api.react_context import ReActContextBuilder
from ops_api.react_memory import IterationRecord

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

    async def create_task(self, session_id: str, task_description: str, max_iterations: int, session, approval_policy: str = "manual") -> str:
        """Create new task"""
        task_id = str(uuid.uuid4())
        task = ReActTask(
            task_id=task_id,
            session_id=session_id,
            task_description=task_description,
            max_iterations=max_iterations,
            session=session
        )
        task.approval_policy = approval_policy

        async with self.lock:
            self.tasks[task_id] = task

        logger.info(f"[TaskManager] Created task {task_id} for session {session_id} with policy {approval_policy}")
        return task_id

    async def start_task(self, task_id: str):
        """Start task"""
        task = self.tasks.get(task_id)
        if not task:
            logger.error(f"[TaskManager] Task {task_id} not found")
            return

        task.status = "running"
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
                max_context_iterations=3
            )

            # Use session model and RAG settings
            model = task.session.model
            rag_enabled = task.session.rag_enabled

            logger.info(f"[TaskManager] Model: {model}, RAG enabled: {rag_enabled}")

            # Execute iteration loop
            while not task.should_stop and task.current_iteration < task.max_iterations:
                task.current_iteration += 1
                iteration_num = task.current_iteration

                logger.info(f"[TaskManager] Iteration {iteration_num}/{task.max_iterations}")

                # Fetch latest image from server
                image_server_client = ImageServerClient()
                image_path = image_server_client.get_last_image()

                # Check if need to stop
                if task.should_stop:
                    logger.info(f"[TaskManager] Task stop requested after fetching image")
                    task.stop()
                    return

                if image_path and not image_path.startswith("./images"):
                    logger.error(f"[TaskManager] Failed to get image: {image_path}")
                    task.error(f"Failed to get image from server: {image_path}")
                    return

                logger.info(f"[TaskManager] Got image: {image_path}")

                # Build prompt
                base_prompt = task.task_description

                # Use memory to build enhanced prompt
                enhanced_prompt = context_builder.build_enhanced_prompt(
                    base_prompt,
                    iteration_num
                )

                # Get API response
                api_client = LLMAPIClient(task.session.api_url, model)
                response = api_client.get_response(
                    enhanced_prompt,
                    image_path=image_path,
                    retrieved_docs=task.session.retriever.retrieve(enhanced_prompt)
                    if rag_enabled and task.session.retriever else None
                )

                logger.info(f"[TaskManager] LLM Response: {response[:200]}...")

                # Parse response
                parser = ResponseParser()
                action, element, input_content, key_content = parser.extract_action_and_element(response)

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
                        task.error(f"Action rejected by user: {task.approval_history[-1].get('reason', 'No reason')}")
                        return
                    elif approval_result == "timeout":
                        logger.info(f"[TaskManager] Approval timeout, stopping task")
                        task.error("Approval timeout")
                        return
                    elif approval_result == "approved":
                        logger.info(f"[TaskManager] Action approved, continuing execution")

                # Process UI actions
                execution_success = False
                execution_result = None
                execution_error = None

                if action in ["Click", "Double Click", "Right Click"] and element and image_path:
                    logger.info(f"[TaskManager] Executing {action} on {element}")
                    executor = CommandExecutor(
                        task.session.ui_model_api_url,
                        task.session.ui_model
                    )
                    success, result = executor.process_ui_element_request(
                        image_path, element, action, element
                    )
                    if success:
                        logger.info(f"[TaskManager] UI action successful: {result}")
                        execution_success = True
                        execution_result = str(result)
                        if isinstance(result, str) and os.path.exists(result):
                            task.iteration_images.append(image_to_base64(result))
                    else:
                        logger.error(f"[TaskManager] UI action failed: {result}")
                        execution_error = str(result)
                elif action == "Input" and input_content:
                    logger.info(f"[TaskManager] Sending input: {input_content}")
                    script_command = f'Send "{input_content}"'
                    image_server_client.send_script_command(script_command)
                    logger.info(f"[TaskManager] Input sent successfully")
                    execution_success = True
                    execution_result = "Input sent successfully"
                elif action == "Keyboard" and key_content:
                    logger.info(f"[TaskManager] Sending keyboard key: {key_content}")
                    script_command = f'Send "{{{key_content}}}"'
                    image_server_client.send_script_command(script_command)
                    logger.info(f"[TaskManager] Keyboard command sent successfully")
                    execution_success = True
                    execution_result = "Keyboard command sent successfully"

                # Update iteration record
                iteration_record.execution_success = execution_success
                iteration_record.execution_result = execution_result
                iteration_record.execution_error = execution_error
                task.session.react_memory_store.add_iteration(task.session_id, iteration_record)
                task.session.react_memory_store.update_patterns(task.session_id, iteration_record)

                # Update progress: display operation result
                task.update_progress(
                    iteration=iteration_num,
                    action=action,
                    element=element,
                    key_content=key_content,
                    reasoning=reasoning,
                    task_status=task_status
                )

                # Check if need to stop
                if task.should_stop:
                    logger.info(f"[TaskManager] Task stop requested before waiting")
                    task.stop()
                    return

                # Wait for system to respond
                logger.info(f"[TaskManager] Waiting for system to respond...")

                # Wait in segments, check stop flag every second
                for _ in range(3):
                    if task.should_stop:
                        logger.info(f"[TaskManager] Task stop requested during wait")
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
