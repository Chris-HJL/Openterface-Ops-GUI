"""
UI-Ins API client module
"""
import requests
from typing import Optional
from config import Config
from ..image.encoder import ImageEncoder

class UIInsClient:
    """UI-Ins API client class"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        Initialize UI-Ins API client

        Args:
            api_url: API URL
            model: Model name
        """
        self.api_url = api_url or Config.DEFAULT_UI_MODEL_API_URL
        self.model = model or Config.DEFAULT_UI_MODEL
        self.timeout = 120

    def call_api(self, image_path: str, instruction: str) -> str:
        """
        Call UI-Ins API

        Args:
            image_path: Image file path
            instruction: Instruction text

        Returns:
            API response text
        """
        try:
            # Build request body
            messages = [
                {
                    "role":"system",
                    # "content": "Provide the coordinate of the element in the screenshot. The coordinate should be in the format of [x, y], enclosed in square brackets."
                    "content": "Locate the UI element specified by the user instruction and provide the coordinate in the format of [x, y]."
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": instruction},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ImageEncoder.encode_to_base64(image_path)}"}}
                    ]
                }
            ]

            payload = {
                "messages": messages,
                "max_tokens": 1000,
                "model": self.model
            }

            # Send POST request
            response = requests.post(
                self.api_url,
                json=payload,
                headers={
                    "Content-Type": "application/json"
                },
                timeout=self.timeout
            )

            # Check response status
            if response.status_code == 200:
                data = response.json()
                # Extract response content
                if "choices" in data and len(data["choices"]) > 0:
                    choice = data["choices"][0]
                    if "message" in choice:
                        return choice["message"]["content"].strip()
                    elif "text" in choice:
                        return choice["text"].strip()
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
