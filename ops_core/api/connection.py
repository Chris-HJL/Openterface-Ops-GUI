"""
API连接测试模块
"""
import requests
from typing import Optional

class APIConnectionTester:
    """API连接测试器类"""

    @staticmethod
    def test_connection(api_url: str, timeout: int = 5) -> bool:
        """
        测试API连接

        Args:
            api_url: API URL
            timeout: 超时时间（秒）

        Returns:
            是否连接成功
        """
        try:
            # 尝试获取模型列表
            models_endpoint = api_url.replace("/chat/completions", "/models")
            test_response = requests.get(models_endpoint, timeout=timeout)
            return test_response.status_code == 200
        except Exception:
            return False
