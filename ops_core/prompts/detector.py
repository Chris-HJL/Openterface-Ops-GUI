"""
Scene detector using VLM for automatic scene detection
"""
import os
import re
import logging
from typing import Optional
import requests
from .types import SceneType
from ..image.encoder import ImageEncoder
from config import Config

logger = logging.getLogger(__name__)


class SceneDetector:
    """Detect scene type from image using VLM"""

    DETECTION_PROMPT = """Analyze this screenshot and determine what type of interface it shows.

Respond with ONLY ONE of the following scene types:
- bios: BIOS/UEFI setup interface (usually blue/gray background, text-based menus, keyboard navigation hints)
- windows: Windows desktop, applications, or system interface, usually with a taskbar, start menu, Windows dialogs, Windows logo, Microsoft Edge browser, etc.
- linux: Linux desktop environment or terminal (GNOME, KDE, XFCE, terminal windows)
- os_installation: Operating system installation interface (Windows Setup, Linux installer, partition screens, license agreements, product key input)
- general: Any other interface or cannot determine

Your response should be in the format: <scene>type</scene>
For example: <scene>bios</scene> or <scene>os_installation</scene>"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        self.api_url = api_url or Config.DEFAULT_API_URL
        self.model = model or Config.DEFAULT_MODEL
        self.api_key = api_key or Config.get_api_key()
        self.timeout = 30

    def detect(self, image_path: str) -> SceneType:
        """
        Detect scene type from image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Detected SceneType
        """
        if not os.path.exists(image_path):
            logger.warning(f"[SceneDetector] Image file not found: {image_path}")
            return SceneType.GENERAL

        try:
            logger.debug(f"[SceneDetector] Detecting scene from image: {image_path}")
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.DETECTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ImageEncoder.encode_to_base64(image_path)}"}}
                    ]
                }
            ]

            payload = {
                "messages": messages,
                "max_tokens": 1000,
                "model": self.model,
            }

            response = requests.post(
                self.api_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                timeout=self.timeout
            )

            if response.status_code == 200:
                data = response.json()
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    content = ""
                    if "text" in choice:
                        content = choice["text"].strip()
                    elif "message" in choice:
                        content = choice["message"]["content"].strip()
                    
                    detected_scene = self._parse_scene_type(content)
                    logger.info(f"[SceneDetector] VLM response: '{content[:50]}...' -> Detected: {detected_scene.value}")
                    return detected_scene

            logger.warning(f"[SceneDetector] API returned status {response.status_code}, using GENERAL")
            return SceneType.GENERAL

        except Exception as e:
            logger.error(f"[SceneDetector] Error detecting scene: {e}")
            return SceneType.GENERAL

    def _parse_scene_type(self, response: str) -> SceneType:
        """Parse scene type from VLM response"""
        scene_match = re.search(r'<scene>\s*(\w+)\s*</scene>', response, re.IGNORECASE)
        
        if scene_match:
            scene_str = scene_match.group(1).lower()
            scene_map = {
                "bios": SceneType.BIOS,
                "uefi": SceneType.BIOS,
                "windows": SceneType.WINDOWS,
                "linux": SceneType.LINUX,
                "os_installation": SceneType.OS_INSTALLATION,
                "osinstallation": SceneType.OS_INSTALLATION,
                "installation": SceneType.OS_INSTALLATION,
                "installer": SceneType.OS_INSTALLATION,
                "general": SceneType.GENERAL,
            }
            return scene_map.get(scene_str, SceneType.GENERAL)
        
        response_lower = response.lower()
        if "bios" in response_lower or "uefi" in response_lower:
            return SceneType.BIOS
        elif "os_installation" in response_lower or "installation" in response_lower or "installer" in response_lower:
            return SceneType.OS_INSTALLATION
        elif "windows" in response_lower:
            return SceneType.WINDOWS
        elif "linux" in response_lower:
            return SceneType.LINUX
        
        return SceneType.GENERAL
