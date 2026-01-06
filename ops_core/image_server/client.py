"""
图像服务器客户端模块
"""
import socket
import datetime
import os
from typing import Optional
from config import Config

class ImageServerClient:
    """图像服务器客户端类"""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        timeout: Optional[int] = None,
        output_dir: Optional[str] = None
    ):
        """
        初始化图像服务器客户端

        Args:
            host: 服务器主机
            port: 服务器端口
            timeout: 超时时间（秒）
            output_dir: 输出目录
        """
        self.host = host or Config.IMAGE_SERVER_HOST
        self.port = port or Config.IMAGE_SERVER_PORT
        self.timeout = timeout or Config.IMAGE_SERVER_TIMEOUT
        self.output_dir = output_dir or Config.IMAGES_DIR

    def get_last_image(self) -> str:
        """
        从服务器获取最新图像

        Returns:
            图像文件路径，失败返回错误消息
        """
        try:
            # 创建输出目录
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)

            # 日志函数
            def log(message):
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"[{timestamp}] {message}")

            log(f"Connecting to image server at {self.host}:{self.port}")

            # 创建TCP连接
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(self.timeout)

            # 连接服务器
            client_socket.connect((self.host, self.port))
            log(f"Connected to {self.host}:{self.port}")

            # 发送"lastimage"命令
            command = "lastimage"
            client_socket.send(command.encode('utf-8'))
            log(f"Sent command: {command}")

            # 接收响应
            response = b""
            total_received = 0
            expected_image_size = 0

            # 首先读取头部信息
            while True:
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    response += data
                    total_received += len(data)

                    # 如果接收到图像数据，尝试解析头部
                    if response.startswith(b"IMAGE:") and b'\n' in response:
                        # 解析图像大小
                        header_end = response.find(b'\n')
                        if header_end != -1:
                            image_size_str = response[6:header_end].decode('utf-8')
                            expected_image_size = int(image_size_str)
                            break
                    elif b"ERROR:" in response or b"STATUS:" in response:
                        # 如果是错误或状态响应，立即停止接收
                        break
                except socket.timeout:
                    log("Timeout while receiving data")
                    break
                except Exception as e:
                    log(f"Error receiving data: {str(e)}")
                    break

            # 继续接收剩余的图像数据
            if expected_image_size > 0:
                expected_total = 6 + len(str(expected_image_size)) + 1 + expected_image_size  # 头部 + 换行符 + 图像数据大小
                while len(response) < expected_total and total_received < expected_total + 10000:  # 添加一些容差
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

            # 检查是否是图像数据
            if response.startswith(b"IMAGE:"):
                # 解析图像数据
                try:
                    # 分离图像大小信息和实际图像数据
                    header_end = response.find(b'\n')
                    if header_end != -1:
                        image_size_str = response[6:header_end].decode('utf-8')
                        image_size = int(image_size_str)
                        image_data = response[header_end+1:]

                        # 验证数据完整性
                        if len(image_data) >= image_size:
                            # 生成文件名
                            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                            filename = f"last_image_{timestamp}.jpg"
                            filepath = os.path.join(self.output_dir, filename)

                            # 保存图像到文件
                            with open(filepath, 'wb') as f:
                                f.write(image_data[:image_size])

                            log(f"Image saved to {filepath}")
                            log(f"Image size: {image_size} bytes")

                            # 关闭连接
                            client_socket.close()
                            return filepath

                except ValueError as ve:
                    log(f"Error parsing image data: {str(ve)}")
                except Exception as e:
                    log(f"Error processing image: {str(e)}")

            # 检查是否是错误或状态响应
            if response.startswith(b"ERROR:"):
                error_msg = response[6:].decode('utf-8', errors='ignore').strip()
                return f"Error: {error_msg}"
            elif response.startswith(b"STATUS:"):
                status_msg = response[7:].decode('utf-8', errors='ignore').strip()
                return f"Status: {status_msg}"

            # 关闭连接
            client_socket.close()
            return "Error: Invalid response from server"

        except socket.timeout:
            return "Error: Connection timeout"
        except Exception as e:
            return f"Error: {str(e)}"

    def send_script_command(self, command: str) -> bool:
        """
        发送脚本命令到服务器

        Args:
            command: 脚本命令

        Returns:
            是否成功
        """
        try:
            # 创建TCP连接
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.settimeout(10)

            # 连接服务器
            client_socket.connect((self.host, self.port))

            # 发送命令
            client_socket.send(command.encode('utf-8'))

            # 接收响应
            response = client_socket.recv(4096)
            response_str = response.decode('utf-8', errors='ignore')

            # 关闭连接
            client_socket.close()

            return True

        except Exception as e:
            print(f"Failed to send script command: {str(e)}")
            return False
