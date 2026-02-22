"""
Prompt registry for managing scene-specific prompts
"""
from typing import Dict, Optional
from .types import SceneType
from .loader import PromptLoader


class PromptRegistry:
    """Registry for managing scene-specific prompts"""

    _instance: Optional['PromptRegistry'] = None
    _prompts: Dict[SceneType, str] = {}
    _loaded: bool = False

    def __new__(cls) -> 'PromptRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._loaded:
            self._load_prompts()

    def _load_prompts(self):
        """Load all prompts from configuration files"""
        loader = PromptLoader()
        
        for scene_type in SceneType:
            if scene_type == SceneType.AUTO:
                continue
            prompt = loader.load(scene_type)
            if prompt:
                self._prompts[scene_type] = prompt
        
        self._loaded = True

    def get(self, scene_type: SceneType) -> str:
        """
        Get prompt for a specific scene type
        
        Args:
            scene_type: The scene type
            
        Returns:
            The system prompt for the scene
        """
        return self._prompts.get(scene_type, self._prompts.get(SceneType.GENERAL, ""))

    def register(self, scene_type: SceneType, prompt: str):
        """
        Register a custom prompt for a scene type
        
        Args:
            scene_type: The scene type
            prompt: The system prompt
        """
        self._prompts[scene_type] = prompt

    def reload(self):
        """Reload all prompts from configuration files"""
        self._prompts.clear()
        self._loaded = False
        self._load_prompts()

    @classmethod
    def get_instance(cls) -> 'PromptRegistry':
        """Get the singleton instance"""
        return cls()
