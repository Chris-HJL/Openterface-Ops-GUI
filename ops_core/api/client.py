"""
API client module
"""
import requests
import os
import logging
from typing import Optional, List, Dict, Any, Union
from config import Config
from ..image.encoder import ImageEncoder
from ..prompts import PromptRegistry, SceneType, SceneDetector

logger = logging.getLogger(__name__)

class LLMAPIClient:
    """LLM API client class"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize API client

        Args:
            api_url: API URL
            model: Model name
            api_key: API key
        """
        self.api_url = api_url or Config.DEFAULT_API_URL
        self.model = model or Config.DEFAULT_MODEL
        self.api_key = api_key or Config.get_api_key()
        self.timeout = 120

    def get_response(
        self,
        prompt: str,
        image_path: Optional[str] = None,
        history: Optional[List[Dict[str, Any]]] = None,
        retrieved_docs: Optional[List[str]] = None,
        system_prompt: Optional[str] = None,
        scene_type: Optional[Union[SceneType, str]] = None
    ) -> str:
        """
        Get API response

        Args:
            prompt: User prompt
            image_path: Image file path
            history: Conversation history
            retrieved_docs: Retrieved documents
            system_prompt: System prompt (overrides scene_type if provided)
            scene_type: Scene type for prompt selection (SceneType enum or string)

        Returns:
            API response text
        """
        try:
            # If there are retrieved documents, add to prompt
            enhanced_prompt = prompt
            if retrieved_docs and len(retrieved_docs) > 0:
                retrieved_content = "\n".join([f"[Relevant Document {i+1}]: {doc}" for i, doc in enumerate(retrieved_docs)])
                enhanced_prompt = f"Answer the user's question based on the following relevant documents:\n{retrieved_content}\n\nUser question: {prompt}"

            # Build request body
            messages = []

            # If there is conversation history, add history
            if history is not None and len(history) > 0:
                messages.extend(history)

            # Add current user message
            if image_path and os.path.exists(image_path):
                # Add image content to message
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ImageEncoder.encode_to_base64(image_path)}"}}
                    ]
                })
                # Determine system prompt
                final_system_prompt = system_prompt
                if not final_system_prompt:
                    registry = PromptRegistry.get_instance()
                    if scene_type:
                        if isinstance(scene_type, str):
                            scene_type = SceneType(scene_type.lower())
                        
                        if scene_type == SceneType.AUTO:
                            logger.info(f"[LLMAPIClient] Scene type is AUTO, detecting scene from image...")
                            detector = SceneDetector(self.api_url, self.model, self.api_key)
                            detected_scene = detector.detect(image_path)
                            logger.info(f"[LLMAPIClient] Detected scene: {detected_scene.value}")
                            final_system_prompt = registry.get(detected_scene)
                        else:
                            logger.info(f"[LLMAPIClient] Using scene type: {scene_type.value}")
                            final_system_prompt = registry.get(scene_type)
                    else:
                        logger.info(f"[LLMAPIClient] No scene type specified, using GENERAL")
                        final_system_prompt = registry.get(SceneType.GENERAL)
                
                messages.append({
                    "role": "system",
                    "content": final_system_prompt
                })
            else:
                messages.append({
                    "role": "user",
                    "content": enhanced_prompt
                })

            payload = {
                "messages": messages,
                "max_tokens": 8000,
                "model": self.model,
                # "temperature": 1.0,
                # "top_p": 0.95,
                # "top_k": 20,
                # "presence_penalty": 0.0,
                # "repetition_penalty": 1.0,
            }

            # Send POST request
            response = requests.post(
                self.api_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                timeout=self.timeout
            )

            # Check response status
            if response.status_code == 200:
                data = response.json()
                # Extract response content
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "text" in choice:
                        return choice["text"].strip()
                    elif "message" in choice:
                        return choice["message"]["content"].strip()
                    elif "delta" in choice:
                        return choice["delta"].get("content", "").strip()
                    else:
                        return str(choice)
                else:
                    return "No response from API"
            else:
                return f"API Error: Status {response.status_code}"

        except requests.exceptions.Timeout:
            return "Request timeout"
        except requests.exceptions.ConnectionError:
            return "Connection error"
        except Exception as e:
            return f"Error occurred: {str(e)}"
