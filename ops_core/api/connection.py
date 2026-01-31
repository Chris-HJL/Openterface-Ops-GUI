"""
API connection test module
"""
import requests
from typing import Optional

class APIConnectionTester:
    """API connection tester class"""

    @staticmethod
    def test_connection(api_url: str, timeout: int = 5) -> bool:
        """
        Test API connection

        Args:
            api_url: API URL
            timeout: Timeout in seconds

        Returns:
            Whether connection was successful
        """
        try:
            # Try to get model list
            models_endpoint = api_url.replace("/chat/completions", "/models")
            test_response = requests.get(models_endpoint, timeout=timeout)
            return test_response.status_code == 200
        except Exception:
            return False
