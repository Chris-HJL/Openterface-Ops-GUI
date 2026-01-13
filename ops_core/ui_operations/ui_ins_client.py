"""
UI-Ins API客户端模块
"""
import requests
from typing import Optional
from config import Config
from ..image.encoder import ImageEncoder

class UIInsClient:
    """UI-Ins API客户端类"""

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: Optional[str] = None
    ):
        """
        初始化UI-Ins API客户端

        Args:
            api_url: API URL
            model: 模型名称
        """
        self.api_url = api_url or Config.DEFAULT_UI_MODEL_API_URL
        self.model = model or Config.DEFAULT_UI_MODEL
        self.timeout = 120

    def call_api(self, image_path: str, instruction: str) -> str:
        """
        调用UI-Ins API

        Args:
            image_path: 图像文件路径
            instruction: 指令文本

        Returns:
            API响应文本
        """
        try:
            # 构建请求体
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
                "max_tokens": 500,
                "model": self.model
            }

            # 发送POST请求
            response = requests.post(
                self.api_url,
                json=payload,
                headers={
                    "Content-Type": "application/json"
                },
                timeout=self.timeout
            )

            # 检查响应状态
            if response.status_code == 200:
                data = response.json()
                # 提取响应内容
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
