"""
Session management module
"""
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from ops_core import Translator, DocumentRetriever, ImageServerClient, LLMAPIClient
from ops_core.ui_operations import CommandExecutor
from ops_core.prompts import SceneType
from config import Config
from .react_memory import ReActMemory, ReActMemoryStore

class Session:
    """Session class for managing each client's state"""

    def __init__(self, session_id: str):
        self.session_id = session_id
        self.conversation_history: List[Dict[str, Any]] = []
        self.is_multiturn_mode = False
        self.current_language = Config.DEFAULT_LANGUAGE
        self.translator = Translator(self.current_language)
        self.retriever: Optional[DocumentRetriever] = None
        self.rag_enabled = False
        # Scene type for prompt selection
        self.scene_type: SceneType = SceneType.AUTO
        # Cached detected scene for ReAct mode (detected once per task)
        self.react_detected_scene: Optional[SceneType] = None
        # API configuration
        self.api_url = Config.DEFAULT_API_URL
        self.model = Config.DEFAULT_MODEL
        self.ui_model_api_url = Config.DEFAULT_UI_MODEL_API_URL
        self.ui_model = Config.DEFAULT_UI_MODEL
        # Current image path
        self.current_image_path: Optional[str] = None
        # ReAct agent configuration
        self.react_enabled = False
        self.react_max_iterations = Config.MAX_REACT_ITERATIONS
        self.react_current_iteration = 0
        self.react_task_description: Optional[str] = None
        self.react_is_running = False

        # ReAct memory system
        self.react_memory: Optional[ReActMemory] = None
        self.react_memory_store = ReActMemoryStore()

        # Timestamps for cleanup
        self.created_at = datetime.now()
        self.updated_at = datetime.now()

        # Cached client instances
        self._image_server_client: Optional[ImageServerClient] = None
        self._llm_api_client: Optional[LLMAPIClient] = None
        self._command_executor: Optional[CommandExecutor] = None

    @property
    def image_server_client(self) -> ImageServerClient:
        """Get or create ImageServerClient instance"""
        if self._image_server_client is None:
            self._image_server_client = ImageServerClient()
        return self._image_server_client

    def get_llm_api_client(self, api_url: Optional[str] = None, model: Optional[str] = None) -> LLMAPIClient:
        """Get or create LLMAPIClient instance"""
        effective_api_url = api_url or self.api_url
        effective_model = model or self.model

        if self._llm_api_client is None:
            self._llm_api_client = LLMAPIClient(effective_api_url, effective_model)
        elif effective_api_url != self._llm_api_client.api_url or effective_model != self._llm_api_client.model:
            self._llm_api_client = LLMAPIClient(effective_api_url, effective_model)

        return self._llm_api_client

    def get_command_executor(self) -> CommandExecutor:
        """Get or create CommandExecutor instance"""
        if self._command_executor is None:
            self._command_executor = CommandExecutor(
                self.ui_model_api_url,
                self.ui_model
            )
        return self._command_executor

    def invalidate_clients(self):
        """Invalidate cached client instances"""
        self._llm_api_client = None
        self._command_executor = None

    def touch(self):
        """Update the last activity timestamp"""
        self.updated_at = datetime.now()

    def switch_language(self, lang_code: str) -> bool:
        """Switch language"""
        if self.translator.switch_language(lang_code):
            self.current_language = lang_code
            return True
        return False

    def switch_scene(self, scene_type: Union[SceneType, str]) -> bool:
        """
        Switch scene type
        
        Args:
            scene_type: SceneType enum or string value
            
        Returns:
            True if switch was successful
        """
        try:
            if isinstance(scene_type, str):
                scene_type = SceneType(scene_type.lower())
            self.scene_type = scene_type
            return True
        except ValueError:
            return False

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []

    def add_to_history(self, role: str, content: Any):
        """Add to conversation history"""
        self.conversation_history.append({"role": role, "content": content})

    def initialize_react_memory(self, task: str):
        """Initialize ReAct memory"""
        self.react_memory = self.react_memory_store.create_memory(
            self.session_id,
            task
        )

    def get_react_memory(self) -> Optional[ReActMemory]:
        """Get ReAct memory"""
        return self.react_memory

    def clear_react_memory(self):
        """Clear ReAct memory"""
        self.react_memory = None
