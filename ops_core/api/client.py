"""
API client module
"""
import requests
import os
from typing import Optional, List, Dict, Any
from config import Config
from ..image.encoder import ImageEncoder

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
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Get API response

        Args:
            prompt: User prompt
            image_path: Image file path
            history: Conversation history
            retrieved_docs: Retrieved documents
            system_prompt: System prompt

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
                # Add system prompt
                if system_prompt:
                    messages.append({
                        "role": "system",
                        "content": system_prompt
                    })
                else:
                    messages.append({
                        "role": "system",
                        "content": """
                            # Language:
                                Use English.
                            # Task:
                                Suggest the next action to perform based on the user instruction/query and the UI elements shown in the image.
                            # Available Actions:
                                - Click
                                - Double Click
                                - Right Click
                                - Input
                                - Keyboard
                            # Output Format:
                                - Enclose your response with <Answer>Your response</Answer>.
                                - Enclose the action to perform with <action></action>, for example, <action>Click</action>, <action>Keyboard</action>.
                                - If the action is Click or Double Click or Right Click, enclose the element name with <element></element> against which the action should be performed, with brief description of the element, for example, <element>browser icon looks like 'e' in the taskbar</element>.
                                - If the action is Input, element tag is not required, you need to trigger a focus/activate event on the text box first by clicking on it, then enclose the text (English only) to input with <input></input>, for example, <input>Hello World</input>.
                                - If the action is Keyboard, element tag is not required, you need to enclose the key to press with <key></key>, for example, <key>Left</key>, <key>Enter</key>.
                                - Enclose the brief explanation of what needs to be done next with <reasoning></reasoning>, for example, <reasoning>Click the OK Button to confirm</reasoning>.
                            # Scenarios:
                                ## BIOS
                                    - Only use Keyboard action, unless there is a mouse cursor shown in the image.
                                    - When only one menu/tab/screen is visible, for example, the 'chipset' screen, and the item desired is not in the current screen, use <key>Esc</key> to exit current screen and back to screen selection.
                                    - Usually <key>Enter</key> is used to expand or confirm or select an item.
                                    - Navigate to the desired item before using <key>Enter</key> to select/expand it.
                                ## Windows and Linux with GUI
                                    - To open an app with icon on desktop or in a folder window, use <action>Double Click</action> on the icon. To open an app with icon in Start Menu or Taskbar, use <action>Click</action> on the icon.
                                    - Before using Input action against a text box, click on the text box to focus/activate it.
                                    - In Windows, copy and paste actions are better to be performed with right-click context menu. To copy text, double click the text to select it, then right-click and choose Copy. To paste text, right-click in the desired location and choose Paste.
                                    - To scroll up or down, use <action>Keyboard</action> with <key>PgUp</key> or <key>PgDn</key> key.
                                    - Read instructions on the screen, fill in the required information before next step.
                                    - Windows OS installation:
                                      * Before installation, when you are required to select the drive/partition to install Windows, format the drive/partition before installation even if its capacity is sufficient.
                                      * When the screen displays something like 'press any key to boot from USB', don't press any key, just wait for the computer to boot normally, then continue with Windows installation.
                        """
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
