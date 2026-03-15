"""
Image server client module
"""
import socket
import datetime
import os
import json
import base64
import time
from typing import Optional, Tuple, Dict, List
from config import Config


class ImageServerClient:
    """Image server client class"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize image server client

        Args:
            host: Server host
            port: Server port
            timeout: Timeout in seconds
            output_dir: Output directory
        """
        self.host = host or Config.IMAGE_SERVER_HOST
        self.port = port or Config.IMAGE_SERVER_PORT
        self.timeout = timeout or Config.SCREEN_CAPTURE_TIMEOUT
        self.output_dir = output_dir or Config.IMAGES_DIR

    def get_target_screen(self) -> str:
        """
        使用 gettargetscreen 命令获取实时屏幕截图

        Returns:
            Image file path (or error message)
        """
        try:
            # 1. 创建输出目录
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # 2. 建立 TCP 连接
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(Config.SCREEN_CAPTURE_TIMEOUT)
            client_socket.connect((self.host, self.port))

            # 3. 发送 gettargetscreen 命令
            client_socket.send(b"gettargetscreen\n")

            # 4. 接收完整响应
            # TCP 是流式协议，需要循环接收直到 JSON 完整
            response_data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk

                # 尝试解析 JSON，如果成功说明数据已完整
                try:
                    json.loads(response_data.decode('utf-8'))
                    break  # JSON 解析成功，退出循环
                except json.JSONDecodeError:
                    continue  # 数据不完整，继续接收

            client_socket.close()

            # 5. 解析 JSON 响应
            response = json.loads(response_data.decode('utf-8'))

            # 6. 检查响应类型和状态
            # 注意：错误响应的 type 是 "error",成功响应的 type 是 "screen"
            if response.get('type') == 'error':
                error_msg = response.get('message', 'Unknown error')
                return f"Error: {error_msg}"

            if response.get('type') != 'screen':
                return f"Error: Unexpected response type: {response.get('type')}"

            if response.get('status') != 'success':
                error_msg = response.get('message', 'Unknown error')
                return f"Error: {error_msg}"

            # 7. 提取图像数据
            data = response.get('data', {})
            base64_content = data.get('content', '')

            if not base64_content:
                return "Error: No image content in response"

            # 8. Base64 解码
            try:
                image_data = base64.b64decode(base64_content)
            except base64.binascii.Error as e:
                return f"Error: Base64 decode failed - {str(e)}"

            # 9. 验证数据大小 (可选)
            expected_size = data.get('size', 0)
            if expected_size and len(base64_content) != expected_size:
                print(f"Warning: Size mismatch. Expected: {expected_size}, Got: {len(base64_content)}")

            # 10. 保存图像文件
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"target_screen_{timestamp}.jpg"
            filepath = os.path.join(self.output_dir, filename)

            with open(filepath, 'wb') as f:
                f.write(image_data)

            # 11. 记录元数据
            if Config.RECORD_SCREEN_RESOLUTION:
                log_msg = (
                    f"Image captured: {filename}, "
                    f"Size: {data.get('size', 0)} bytes, "
                    f"Resolution: {data.get('width', 0)}x{data.get('height', 0)}"
                )
                print(log_msg)

            return filepath

        except json.JSONDecodeError as e:
            return f"Error: Invalid JSON response - {str(e)}"
        except socket.timeout:
            return "Error: Connection timeout"
        except Exception as e:
            return f"Error: {str(e)}"

    def send_script_command(self, command: str) -> bool:
        """
        Send script command to server

        Args:
            command: Script command

        Returns:
            Whether successful
        """
        try:
            # Create TCP connection
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(10)

            # Connect to server
            client_socket.connect((self.host, self.port))

            # Send command (with newline to match TCP server protocol)
            client_socket.send((command + '\n').encode('utf-8'))

            # Receive response
            response = client_socket.recv(4096)
            response_str = response.decode('utf-8', errors='ignore')

            # Close connection
            client_socket.close()

            return True

        except Exception as e:
            print(f"Failed to send script command: {str(e)}")
            return False

    def send_script_command_with_response(
        self,
        command: str,
        keep_connection: bool = False
    ) -> Tuple[bool, str]:
        """
        发送脚本命令并返回响应

        Args:
            command: 脚本命令（如 'Click 2048,2048' 或 'Send "text"'）
            keep_connection: 是否保持连接（用于后续 checkstatus）

        Returns:
            (success, response_text)
            - success: 发送是否成功
            - response_text: 服务器响应内容（可能为空）

        Note:
            当前 TCP 服务器实现中，send_script_command 不返回执行结果
            此方法用于未来扩展或支持状态检查的场景

        Example:
            >>> client = ImageServerClient()
            >>> success, response = client.send_script_command_with_response('Click 2048,2048')
            >>> print(f"Success: {success}, Response: {response}")
        """
        try:
            # 建立 TCP 连接
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(10)
            client_socket.connect((self.host, self.port))

            # 发送命令（添加换行符，符合 TCP 服务器协议）
            client_socket.send((command + '\n').encode('utf-8'))

            # 接收响应（如果有）
            response = b""
            if not keep_connection:
                # 不保持连接时，尝试接收响应
                client_socket.settimeout(2)
                try:
                    response = client_socket.recv(4096)
                except socket.timeout:
                    response = b""

            response_text = response.decode('utf-8', errors='ignore')

            # 关闭连接（如果不需要保持）
            if not keep_connection:
                client_socket.close()

            return (True, response_text)

        except Exception as e:
            print(f"Failed to send script command: {str(e)}")
            return (False, str(e))

    def send_command_sequence(
        self,
        commands: List[str],
        delay: float = 0.5,
        use_persistent_connection: bool = True
    ) -> bool:
        """
        批量发送命令序列

        Args:
            commands: 命令列表，如 ['Click 2048,2048', 'Send "username"']
            delay: 命令间延迟（秒），默认 0.5
            use_persistent_connection: 是否使用持久连接（推荐，提高性能）

        Returns:
            是否全部发送成功

        Note:
            使用持久连接可以：
            1. 减少连接建立开销
            2. 支持命令间状态检查
            3. 提高执行速度

        Example:
            >>> client = ImageServerClient()
            >>> success = client.send_command_sequence([
            ...     'Click 2048,2048',
            ...     'Send "username"',
            ...     'Send "{Tab}"',
            ...     'Send "password123"'
            ... ], delay=0.5, use_persistent_connection=True)
        """
        try:
            # 建立持久连接
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(10)
            client_socket.connect((self.host, self.port))

            for i, command in enumerate(commands):
                # 发送命令（添加换行符，符合 TCP 服务器协议）
                client_socket.send((command + '\n').encode('utf-8'))

                # 短暂等待，让服务器处理
                if not use_persistent_connection:
                    # 非持久模式，每次命令后尝试接收响应
                    client_socket.settimeout(2)
                    try:
                        response = client_socket.recv(4096)
                    except socket.timeout:
                        pass

                # 如果不是最后一个命令，等待延迟
                if i < len(commands) - 1:
                    time.sleep(delay)

            # 关闭连接
            client_socket.close()

            return True

        except Exception as e:
            print(f"[ImageServerClient] Failed to send command sequence: {e}")
            return False

    def send_command_sequence_with_check(
        self,
        commands: List[str],
        delay: float = 0.5,
        timeout: int = 30
    ) -> List[Dict]:
        """
        批量发送命令序列并检查状态

        Args:
            commands: 命令列表
            delay: 命令间延迟（秒）
            timeout: 状态检查超时（秒）

        Returns:
            [
                {
                    "index": 0,
                    "command": "Click 2048,2048",
                    "sent": True,
                    "status": "finish",  # or "running" / "failed"
                    "message": "Command execution completed successfully"
                },
                ...
            ]

        Note:
            由于 TCP 服务器的 checkstatus 只能检查最后一个命令，
            此方法会在发送每个命令后检查状态。

            使用持久连接以提高性能和准确性。

        Example:
            >>> results = client.send_command_sequence_with_check([
            ...     'Click 2048,2048',
            ...     'Send "{F5}"'
            ... ])
            >>> for r in results:
            ...     print(f"Command {r['index']}: {r['status']}")
        """
        results = []

        try:
            # 建立持久连接
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(timeout)
            client_socket.connect((self.host, self.port))

            for i, command in enumerate(commands):
                result = {
                    "index": i,
                    "command": command,
                    "sent": False,
                    "status": None,
                    "message": None
                }

                # 发送命令（添加换行符，符合 TCP 服务器协议）
                try:
                    client_socket.send((command + '\n').encode('utf-8'))
                    result["sent"] = True
                except Exception as e:
                    result["status"] = "failed"
                    result["message"] = f"Failed to send: {str(e)}"
                    results.append(result)
                    continue

                # 发送 checkstatus 命令（添加换行符）
                try:
                    client_socket.send(b"checkstatus\n")

                    # 接收状态响应
                    client_socket.settimeout(5)
                    response_data = b""
                    while True:
                        chunk = client_socket.recv(4096)
                        if not chunk:
                            break
                        response_data += chunk

                        # 尝试解析 JSON
                        try:
                            response = json.loads(response_data.decode('utf-8'))
                            if response.get('type') == 'status':
                                result["status"] = response.get('status', 'failed')
                                result["message"] = response.get('message', '')
                                break
                        except json.JSONDecodeError:
                            continue

                    if result["status"] is None:
                        result["status"] = "unknown"
                        result["message"] = "No status response received"

                except Exception as e:
                    result["status"] = "unknown"
                    result["message"] = f"Status check failed: {str(e)}"

                results.append(result)

                # 等待延迟
                if i < len(commands) - 1:
                    time.sleep(delay)

            # 关闭连接
            client_socket.close()

        except Exception as e:
            print(f"[ImageServerClient] Error in sequence with check: {e}")
            # 添加错误结果
            results.append({
                "index": len(commands),
                "command": "ERROR",
                "sent": False,
                "status": "failed",
                "message": str(e)
            })

        return results

    def check_command_status(self) -> str:
        """
        检查命令执行状态

        Returns:
            "success" | "running" | "failed"

        Note:
            仅能检查最后一个命令的状态
            需要 TCP 服务器支持 checkstatus 命令

        Example:
            >>> client.send_script_command('Send "{F5}"')
            >>> status = client.check_command_status()
            >>> print(status)
            "success"
        """
        try:
            # 建立连接
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5)
            client_socket.connect((self.host, self.port))

            # 发送 checkstatus 命令（添加换行符，符合 TCP 服务器协议）
            client_socket.send(b"checkstatus\n")

            # 接收响应
            response_data = b""
            while True:
                chunk = client_socket.recv(4096)
                if not chunk:
                    break
                response_data += chunk

                # 尝试解析 JSON
                try:
                    response = json.loads(response_data.decode('utf-8'))
                    if response.get('type') == 'status':
                        client_socket.close()
                        status = response.get('status', 'failed')
                        # 标准化状态值
                        if status in ['finish', 'success']:
                            return "success"
                        elif status in ['fail', 'failed']:
                            return "failed"
                        else:
                            return status
                except json.JSONDecodeError:
                    continue

            client_socket.close()
            return "failed"

        except Exception as e:
            print(f"[ImageServerClient] Error checking status: {e}")
            return "failed"

    def reconnect(self):
        """
        重新建立 TCP 连接

        用于执行前确保连接有效
        当前实现仅测试连接，不保持连接
        """
        try:
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(5)
            client_socket.connect((self.host, self.port))
            client_socket.close()
            return True
        except Exception as e:
            print(f"[ImageServerClient] Reconnect failed: {e}")
            return False
