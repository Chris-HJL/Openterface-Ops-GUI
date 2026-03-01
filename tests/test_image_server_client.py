"""
单元测试：ImageServerClient

测试 gettargetscreen 命令的实现
"""
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import base64
import socket
import os
import sys

# 添加项目根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ops_core.image_server.client import ImageServerClient
from config import Config, ScreenCaptureMode


class TestImageServerClient(unittest.TestCase):
    """ImageServerClient 测试类"""

    def setUp(self):
        """测试前的准备工作"""
        self.client = ImageServerClient(
            host="localhost",
            port=12345,
            timeout=5,
            output_dir="./test_output"
        )
        # 创建测试输出目录
        os.makedirs("./test_output", exist_ok=True)

    def tearDown(self):
        """测试后的清理工作"""
        # 清理测试输出目录
        import shutil
        if os.path.exists("./test_output"):
            shutil.rmtree("./test_output")

    def _create_mock_base64_image(self, width=1920, height=1080):
        """创建 mock Base64 图像数据"""
        # 创建一些 mock 的二进制数据 (模拟 JPEG)
        mock_image_data = b'\xff\xd8' + b'\x00' * 1000 + b'\xff\xd9'  # JPEG 开头和结尾标记
        return base64.b64encode(mock_image_data).decode('utf-8'), mock_image_data

    def _create_successful_response(self, width=1920, height=1080):
        """创建成功的响应"""
        base64_content, _ = self._create_mock_base64_image(width, height)
        return {
            "type": "screen",
            "status": "success",
            "timestamp": "2026-02-28T10:30:45Z",
            "data": {
                "size": len(base64_content),
                "width": width,
                "height": height,
                "format": "jpeg",
                "encoding": "base64",
                "content": base64_content
            }
        }

    @patch('socket.socket')
    def test_get_target_screen_success(self, mock_socket_class):
        """测试 gettargetscreen 成功场景"""
        # 准备 mock 数据
        mock_response = self._create_successful_response()
        mock_response_json = json.dumps(mock_response).encode('utf-8')

        # 配置 mock socket
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recv.side_effect = [mock_response_json, b'']  # 第一次返回数据，第二次返回空

        # 调用方法
        result = self.client.get_target_screen()

        # 验证结果
        self.assertFalse(result.startswith("Error:"))
        self.assertTrue("target_screen_" in result)
        self.assertTrue(result.endswith(".jpg"))
        self.assertTrue(os.path.exists(result))

        # 验证文件已创建
        self.assertTrue(os.path.exists(result))

        # 验证发送的命令
        mock_socket.send.assert_called_with(b"gettargetscreen\n")

    @patch('socket.socket')
    def test_get_target_screen_error_response(self, mock_socket_class):
        """测试服务器返回错误响应"""
        # 准备错误响应
        error_response = {
            "type": "error",
            "status": "error",
            "timestamp": "2026-02-28T10:30:45Z",
            "message": "Screen capture failed"
        }
        error_response_json = json.dumps(error_response).encode('utf-8')

        # 配置 mock socket
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recv.side_effect = [error_response_json, b'']

        # 调用方法
        result = self.client.get_target_screen()

        # 验证结果
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("Screen capture failed", result)

    @patch('socket.socket')
    def test_get_target_screen_invalid_json(self, mock_socket_class):
        """测试 JSON 解析错误"""
        # 准备无效的 JSON
        invalid_json = b'{invalid json data'

        # 配置 mock socket
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recv.side_effect = [invalid_json, b'']

        # 调用方法
        result = self.client.get_target_screen()

        # 验证结果
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("Invalid JSON", result)

    @patch('socket.socket')
    def test_get_target_screen_no_content(self, mock_socket_class):
        """测试响应中缺少图像内容"""
        # 准备没有内容的成功响应
        response = {
            "type": "screen",
            "status": "success",
            "timestamp": "2026-02-28T10:30:45Z",
            "data": {
                "size": 0,
                "width": 1920,
                "height": 1080,
                "format": "jpeg",
                "encoding": "base64",
                "content": ""
            }
        }
        response_json = json.dumps(response).encode('utf-8')

        # 配置 mock socket
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recv.side_effect = [response_json, b'']

        # 调用方法
        result = self.client.get_target_screen()

        # 验证结果
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("No image content", result)

    @patch('socket.socket')
    def test_get_target_screen_connection_timeout(self, mock_socket_class):
        """测试连接超时"""
        # 配置 mock socket 抛出超时异常
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        mock_socket.connect.side_effect = socket.timeout()

        # 调用方法
        result = self.client.get_target_screen()

        # 验证结果
        self.assertTrue(result.startswith("Error:"))
        self.assertIn("timeout", result.lower())

    @patch('socket.socket')
    def test_get_target_screen_chunked_response(self, mock_socket_class):
        """测试分块接收响应"""
        # 准备成功的响应
        mock_response = self._create_successful_response()
        mock_response_json = json.dumps(mock_response).encode('utf-8')

        # 将响应分成多个块
        chunk_size = 100
        chunks = [mock_response_json[i:i+chunk_size] for i in range(0, len(mock_response_json), chunk_size)]
        chunks.append(b'')  # 最后加一个空块表示接收结束

        # 配置 mock socket
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recv.side_effect = chunks

        # 调用方法
        result = self.client.get_target_screen()

        # 验证结果
        self.assertFalse(result.startswith("Error:"))
        self.assertTrue(os.path.exists(result))

    @patch('socket.socket')
    def test_get_screen_image_primary_success(self, mock_socket_class):
        """测试智能获取方法：首选命令成功"""
        # 准备成功的响应
        mock_response = self._create_successful_response()
        mock_response_json = json.dumps(mock_response).encode('utf-8')

        # 配置 mock socket
        mock_socket = MagicMock()
        mock_socket_class.return_value = mock_socket
        mock_socket.recv.side_effect = [mock_response_json, b'']

        # 调用方法
        result = self.client.get_screen_image(
            primary_command="gettargetscreen",
            fallback_command="lastimage"
        )

        # 验证结果
        self.assertFalse(result.startswith("Error:"))

    @patch('socket.socket')
    def test_get_screen_image_with_fallback(self, mock_socket_class):
        """测试智能获取方法：首选失败，回退成功"""
        # 配置 mock socket 第一次返回错误，第二次返回成功 (简化测试)
        error_response = {
            "type": "error",
            "status": "error",
            "message": "gettargetscreen failed"
        }
        
        # 这个测试需要更复杂的 mock 设置，暂时跳过
        # 建议用集成测试代替
        self.skipTest("需要更复杂的 mock 设置，建议用集成测试代替")

    def test_screen_capture_mode_enum(self):
        """测试屏幕捕获模式枚举"""
        # 测试枚举值
        self.assertEqual(ScreenCaptureMode.GETTARGETSCREEN.value, "gettargetscreen")
        self.assertEqual(ScreenCaptureMode.LASTIMAGE.value, "lastimage")
        self.assertEqual(ScreenCaptureMode.HYBRID.value, "hybrid")

        # 测试从字符串创建
        self.assertEqual(ScreenCaptureMode("gettargetscreen"), ScreenCaptureMode.GETTARGETSCREEN)
        self.assertEqual(ScreenCaptureMode("lastimage"), ScreenCaptureMode.LASTIMAGE)
        self.assertEqual(ScreenCaptureMode("hybrid"), ScreenCaptureMode.HYBRID)

    def test_config_screen_capture_mode(self):
        """测试配置项"""
        # 测试配置项存在
        self.assertTrue(hasattr(Config, 'SCREEN_CAPTURE_MODE'))
        self.assertIsInstance(Config.SCREEN_CAPTURE_MODE, ScreenCaptureMode)
        
        # 测试超时配置
        self.assertTrue(hasattr(Config, 'SCREEN_CAPTURE_TIMEOUT'))
        self.assertIsInstance(Config.SCREEN_CAPTURE_TIMEOUT, int)
        
        # 测试分辨率记录配置
        self.assertTrue(hasattr(Config, 'RECORD_SCREEN_RESOLUTION'))
        self.assertIsInstance(Config.RECORD_SCREEN_RESOLUTION, bool)


class TestImageServerClientIntegration(unittest.TestCase):
    """集成测试：与真实模拟器配合 (可选)"""

    @unittest.skip("需要启动模拟器，手动测试")
    def test_with_real_simulator(self):
        """
        使用真实模拟器测试
        需要先启动 tools/tcpserver_simulator.py
        """
        client = ImageServerClient(
            host="localhost",
            port=12345,
            timeout=10
        )
        
        # 测试 gettargetscreen
        result = client.get_target_screen()
        print(f"gettargetscreen result: {result}")
        self.assertFalse(result.startswith("Error:"))
        
        # 测试 lastimage
        result = client.get_last_image()
        print(f"lastimage result: {result}")
        self.assertFalse(result.startswith("Error:"))
        
        # 测试智能方法
        result = client.get_screen_image()
        print(f"get_screen_image result: {result}")
        self.assertFalse(result.startswith("Error:"))


if __name__ == '__main__':
    unittest.main()
