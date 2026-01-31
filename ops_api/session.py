"""
Session management module
"""
from typing import Dict, Any, Optional, List
from ops_core import Translator, DocumentRetriever
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

    def switch_language(self, lang_code: str) -> bool:
        """Switch language"""
        if self.translator.switch_language(lang_code):
            self.current_language = lang_code
            return True
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
