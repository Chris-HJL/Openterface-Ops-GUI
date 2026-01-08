"""
API客户端模块
"""
import requests
import os
from typing import Optional, List, Dict, Any
from config import Config
from ..image.encoder import ImageEncoder

class LLMAPIClient:
    """LLM API客户端类"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """
        初始化API客户端

        Args:
            api_url: API URL
            model: 模型名称
            api_key: API密钥
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
        获取API响应

        Args:
            prompt: 用户提示
            image_path: 图像文件路径
            history: 对话历史
            retrieved_docs: 检索到的文档
            system_prompt: 系统提示

        Returns:
            API响应文本
        """
        try:
            # 如果有检索到的文档，添加到提示中
            enhanced_prompt = prompt
            if retrieved_docs and len(retrieved_docs) > 0:
                retrieved_content = "\n".join([f"[Relevant Document {i+1}]: {doc}" for i, doc in enumerate(retrieved_docs)])
                enhanced_prompt = f"Answer the user's question based on the following relevant documents:\n{retrieved_content}\n\nUser question: {prompt}"

            # 构建请求体
            messages = []

            # 如果有对话历史，添加历史
            if history is not None and len(history) > 0:
                messages.extend(history)

            # 添加当前用户消息
            if image_path and os.path.exists(image_path):
                # 添加图像内容到消息
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "text", "text": enhanced_prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{ImageEncoder.encode_to_base64(image_path)}"}}
                    ]
                })
                # 添加系统提示
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
                                - Enclose the action to perform with <action></action>, e.g. <action>Click</action>, <action>Keyboard</action>.
                                - Enclose the element name with with brief description <element></element> where the action should be performed, e.g. <element>OK Button</element>.
                                - Enclose the text to input with <input></input> if the action is Input, e.g. <input>Hello World</input>.
                                - Enclose the key to press with <key></key> if the action is Keyboard, e.g. <key>Left</key>, <key>Enter</key>.
                                - Enclose the brief explanation of what needs to be done next with <reasoning></reasoning>, e.g. <reasoning>Click the OK Button to confirm</reasoning>.
                            # Scenario:
                                - If it is in BIOS, only use Keyboard action, unless there is a mouse cursor shown in the image.
                                - To open an app with desktop icon, use <action>Double Click</action> on the icon. To open an app with icon in Start Menu or Taskbar, use <action>Click</action> on the icon.
                                - Before using Input action, make sure the text box is focused. If not, use <action>Click</action> to focus it.
                                - In Windows, copy and paste actions are better to be performed with right-click context menu. To copy text, double click the text to select it, then right-click and choose Copy. To paste text, right-click in the desired location and choose Paste.
                                - Prefer using mouse actions over keyboard actions, unless there is a specific reason to use keyboard actions.
                                - To scroll up or down, use <action>Keyboard</action> with <key>PgUp</key> or <key>PgDn</key> key.
                        """
                    })
            else:
                messages.append({
                    "role": "user",
                    "content": enhanced_prompt
                })

            payload = {
                "messages": messages,
                "max_tokens": 8192,
                "model": self.model
            }

            # 发送POST请求
            response = requests.post(
                self.api_url,
                json=payload,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                timeout=self.timeout
            )

            # 检查响应状态
            if response.status_code == 200:
                data = response.json()
                # 提取响应内容
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
