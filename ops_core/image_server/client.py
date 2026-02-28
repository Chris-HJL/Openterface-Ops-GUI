"""
Image server client module
"""
import socket
import datetime
import os
import json
import base64
from typing import Optional
from config import Config, ScreenCaptureMode

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
        self.timeout = timeout or Config.IMAGE_SERVER_TIMEOUT
        self.output_dir = output_dir or Config.IMAGES_DIR

    def get_last_image(self) -> str:
        """
        Get latest image from server

        Returns:
            Image file path, returns error message on failure
        """
        try:
            # Create output directory
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # Log function
            def log(message):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] {message}")

            log(f"Connecting to image server at {self.host}:{self.port}")

            # Create TCP connection
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(self.timeout)

            # Connect to server
            client_socket.connect((self.host, self.port))
            log(f"Connected to {self.host}:{self.port}")

            # Send "lastimage" command
            command = "lastimage"
            client_socket.send(command.encode('utf-8'))
            log(f"Sent command: {command}")

            # Receive response
            response = b""
            total_received = 0
            expected_image_size = 0

            # Read header information first
            while True:
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    response += data
                    total_received += len(data)

                    # If image data is received, try to parse header
                    if response.startswith(b"IMAGE:") and b'\n' in response:
                        # Parse image size
                        header_end = response.find(b'\n')
                        if header_end != -1:
                            image_size_str = response[6:header_end].decode('utf-8')
                            expected_image_size = int(image_size_str)
                            break
                    elif b"ERROR:" in response or b"STATUS:" in response:
                        # If error or status response, stop receiving immediately
                        break
                except socket.timeout:
                    log("Timeout while receiving data")
                    break
                except Exception as e:
                    log(f"Error receiving data: {str(e)}")
                    break

            # Continue receiving remaining image data
            if expected_image_size > 0:
                expected_total = 6 + len(str(expected_image_size)) + 1 + expected_image_size  # Header + newline + image data size
                while len(response) < expected_total and total_received < expected_total + 10000:  # Add some tolerance
                    try:
                        data = client_socket.recv(min(4096, expected_total - len(response)))
                        if not data:
                            break
                        response += data
                        total_received += len(data)
                    except socket.timeout:
                        log("Timeout while receiving image data")
                        break
                    except Exception as e:
                        log(f"Error receiving image data: {str(e)}")
                        break

            log(f"Received {len(response)} bytes")

            # Check if it's image data
            if response.startswith(b"IMAGE:"):
                # Parse image data
                try:
                    # Separate image size info and actual image data
                    header_end = response.find(b'\n')
                    if header_end != -1:
                        image_size_str = response[6:header_end].decode('utf-8')
                        image_size = int(image_size_str)
                        image_data = response[header_end+1:]

                        # Verify data integrity
                        if len(image_data) >= image_size:
                            # Generate filename
                            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"last_image_{timestamp}.jpg"
                            filepath = os.path.join(self.output_dir, filename)

                            # Save image to file
                            with open(filepath, 'wb') as f:
                                f.write(image_data[:image_size])

                            log(f"Image saved to {filepath}")
                            log(f"Image size: {image_size} bytes")

                            # Close connection
                            client_socket.close()
                            return filepath

                except ValueError as ve:
                    log(f"Error parsing image data: {str(ve)}")
                except Exception as e:
                    log(f"Error processing image: {str(e)}")

            # Check if it's error or status response
            if response.startswith(b"ERROR:"):
                error_msg = response[6:].decode('utf-8', errors='ignore').strip()
                return f"Error: {error_msg}"
            elif response.startswith(b"STATUS:"):
                status_msg = response[7:].decode('utf-8', errors='ignore').strip()
                return f"Status: {status_msg}"

            # Close connection
            client_socket.close()
            return "Error: Invalid response from server"

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

            # Send command
            client_socket.send(command.encode('utf-8'))

            # Receive response
            response = client_socket.recv(4096)
            response_str = response.decode('utf-8', errors='ignore')

            # Close connection
            client_socket.close()

            return True

        except Exception as e:
            print(f"Failed to send script command: {str(e)}")
            return False

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

    def get_screen_image(
        self,
        primary_command: str = "gettargetscreen",
        fallback_command: str = "lastimage"
    ) -> str:
        """
        智能获取屏幕图片，支持命令选择和自动回退

        Args:
            primary_command: 首选命令 (gettargetscreen/lastimage)
            fallback_command: 回退命令

        Returns:
            Image file path or error message
        """
        # 尝试首选命令
        if primary_command == "gettargetscreen":
            result = self.get_target_screen()
        else:
            result = self.get_last_image()

        # 如果首选命令失败，尝试回退命令
        if result.startswith("Error:"):
            print(f"Primary command '{primary_command}' failed: {result}")
            if fallback_command:
                print(f"Falling back to '{fallback_command}'...")
                if fallback_command == "gettargetscreen":
                    result = self.get_target_screen()
                else:
                    result = self.get_last_image()

        return result
