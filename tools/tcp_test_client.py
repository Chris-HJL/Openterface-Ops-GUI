#!/usr/bin/env python3
"""
TCP 屏幕图像获取测试客户端 (gettargetscreen)

本工具用于测试 Openterface Mini-KVM QT 设备的 `gettargetscreen` 命令协议。
参考文档：TCP_SCREEN_CAPTURE_MIGRATION_PLAN.md

主要功能:
  - 连接 TCP 服务器
  - 发送 gettargetscreen 命令
  - 流式接收 JSON 响应 (处理 TCP 粘包/分包)
  - 解码 Base64 图像数据
  - 保存图像到本地文件

Usage:
  # 使用默认配置 (localhost:12345)
  python tools/tcp_test_client.py

  # 自定义服务器地址和端口
  python tools/tcp_test_client.py --host 192.168.1.100 --port 2345

  # 自定义超时时间和输出目录
  python tools/tcp_test_client.py --timeout 60 --output ./my_images
"""

import socket
import json
import base64
import os
import sys
import time
import argparse
from datetime import datetime
from typing import Tuple, Optional


class TargetScreenClient:
    """
    gettargetscreen 命令的测试客户端
    
    实现协议:
      - 发送： "gettargetscreen\n"
      - 接收：流式 JSON (type: screen, status: success/error)
      - 图像：Base64 编码的 JPEG
    """
    
    def __init__(
        self,
        host: str = "localhost",
        port: int = 12345,
        timeout: int = 120,
        output_dir: str = "./output"
    ):
        """
        初始化客户端
        
        Args:
            host: 服务器地址
            port: 服务器端口
            timeout: 连接和接收超时 (秒), gettargetscreen 需要较长时间
            output_dir: 图像保存目录
        """
        self.host = host
        self.port = port
        self.timeout = timeout
        self.output_dir = output_dir
        
        # 确保输出目录存在
        os.makedirs(self.output_dir, exist_ok=True)
    
    def connect(self) -> socket.socket:
        """建立 TCP 连接"""
        print(f"🔄 Connecting to {self.host}:{self.port} (timeout={self.timeout}s)...")
        client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client_socket.settimeout(self.timeout)
        
        try:
            client_socket.connect((self.host, self.port))
            print("✅ Connected successfully")
            return client_socket
        except socket.timeout:
            raise Exception("❌ Connection timeout")
        except ConnectionRefusedError:
            raise Exception(f"❌ Connection refused to {self.host}:{self.port}")
        except Exception as e:
            raise Exception(f"❌ Connection error: {str(e)}")
    
    def send_command(self, client_socket: socket.socket, command: str) -> None:
        """发送命令"""
        command_bytes = f"{command}\n".encode('utf-8')
        print(f"📤 Sending command: '{command}'")
        client_socket.send(command_bytes)
    
    def receive_json_response(self, client_socket: socket.socket) -> str:
        """
        流式接收 JSON 响应
        
        ⚠️ 关键点: TCP 是流式协议，数据可能分多个包到达
        策略：循环接收，直到 buffer 能解析为完整 JSON
        """
        print("📥 Receiving response...")
        buffer = b""
        start_time = time.time()
        
        while True:
            # 1. 检查是否超时
            elapsed = time.time() - start_time
            if elapsed > self.timeout:
                raise Exception(f"❌ Receive timeout after {self.timeout}s")
            
            # 2. 接收数据块
            try:
                chunk = client_socket.recv(4096)
            except socket.timeout:
                # 如果 buffer 已有数据，可能是网络慢，继续尝试
                if buffer:
                    continue
                else:
                    break
            
            if not chunk:
                break  # 连接关闭
            
            buffer += chunk
            
            # 3. 尝试解析 JSON (关键步骤)
            try:
                json_str = buffer.decode('utf-8')
                json.loads(json_str)  # 验证是否合法
                print(f"✅ JSON received and validated (size: {len(json_str)} bytes)")
                return json_str
            except json.JSONDecodeError:
                # JSON 不完整，继续接收
                if len(buffer) > 10 * 1024 * 1024:  # 10MB 保护上限
                    raise Exception("❌ Response buffer exceeds 10MB, possible corruption")
                continue
            
            except UnicodeDecodeError:
                # 可能是二进制垃圾数据
                print(f"⚠️ Unicode decode error, buffer size: {len(buffer)}, continuing...")
                continue
    
    def parse_response(self, json_str: str) -> dict:
        """解析 JSON 响应"""
        try:
            response = json.loads(json_str)
            
            # 验证响应格式
            if not isinstance(response, dict):
                raise ValueError("Response is not a JSON object")
            
            response_type = response.get('type', '')
            status = response.get('status', '')
            
            print(f"📋 Response type: {response_type}, status: {status}")
            
            # 检查是否成功
            if response_type == 'error' or status == 'error':
                message = response.get('message', 'Unknown error')
                raise Exception(f"❌ Server Error: {message}")
            
            if response_type != 'screen':
                raise Exception(f"❌ Unexpected response type: {response_type}")
            
            if status != 'success':
                raise Exception(f"❌ Unexpected status: {status}")
            
            return response
            
        except json.JSONDecodeError as e:
            raise Exception(f"❌ Invalid JSON: {str(e)}")
    
    def decode_and_save_image(self, response: dict) -> str:
        """解码图像并保存"""
        data = response.get('data', {})
        
        if not data:
            raise Exception("❌ No 'data' field in response")
        
        # 提取 Base64 内容
        base64_content = data.get('content', '')
        if not base64_content:
            raise Exception("❌ No 'content' field in data")
        
        print(f"🔐 Decoding Base64 content (size: {len(base64_content)} chars)...")
        
        try:
            image_data = base64.b64decode(base64_content)
        except base64.binascii.Error as e:
            raise Exception(f"❌ Base64 decode failed: {str(e)}")
        
        print(f"✅ Image decoded (size: {len(image_data)} bytes)")
        
        # 验证元数据 (可选)
        if data.get('format', '').lower() not in ['jpeg', 'jpg']:
            print(f"⚠️ Image format is '{data.get('format', 'unknown')}', expecting JPEG")
        
        expected_size = data.get('size', 0)
        if expected_size and len(base64_content) != expected_size:
            print(f"⚠️ Size mismatch in metadata (expected: {expected_size}, got: {len(base64_content)})")
        
        # 保存文件
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"target_screen_{timestamp}.jpg"
        filepath = os.path.join(self.output_dir, filename)
        
        try:
            with open(filepath, 'wb') as f:
                f.write(image_data)
            print(f"💾 Image saved to: {filepath}")
        except IOError as e:
            raise Exception(f"❌ Failed to save image: {str(e)}")
        
        # 打印元数据摘要
        resolution = f"{data.get('width', 'N/A')}x{data.get('height', 'N/A')}"
        print(f"📊 Image Summary:")
        print(f"   Resolution: {resolution}")
        print(f"   Size: {len(image_data):,} bytes ({len(image_data)/1024:.1f} KB)")
        print(f"   Format: {data.get('format', 'jpeg')}")
        print(f"   Timestamp: {response.get('timestamp', 'N/A')}")
        
        return filepath
    
    def fetch_target_screen(self) -> str:
        """
        执行完整的 gettargetscreen 流程
        
        Returns:
            图像文件路径
        """
        client_socket = None
        try:
            # 1. 连接
            client_socket = self.connect()
            
            # 2. 发送命令
            self.send_command(client_socket, "gettargetscreen")
            
            # 3. 接收响应
            json_str = self.receive_json_response(client_socket)
            
            # 4. 解析响应
            response = self.parse_response(json_str)
            
            # 5. 解码并保存
            filepath = self.decode_and_save_image(response)
            
            return filepath
            
        finally:
            # 6. 清理连接
            if client_socket:
                try:
                    client_socket.close()
                    print("🔒 Connection closed")
                except:
                    pass
    
    def run(self) -> int:
        """
        运行客户端
        
        Returns:
            状态码 (0=成功，1=失败)
        """
        print("=" * 60)
        print("TCP Target Screen Test Client")
        print("=" * 60)
        print(f"Host: {self.host}")
        print(f"Port: {self.port}")
        print(f"Timeout: {self.timeout}s")
        print(f"Output: {self.output_dir}")
        print("=" * 60)
        print()
        
        try:
            filepath = self.fetch_target_screen()
            print("\n" + "=" * 60)
            print("✅ Success!")
            print("=" * 60)
            return 0
        except Exception as e:
            print("\n" + "=" * 60)
            print(f"❌ Failed: {str(e)}")
            print("=" * 60)
            return 1


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="Test gettargetscreen TCP command via Openterface Mini-KVM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test with default settings (localhost:12345)
  python tcp_test_client.py

  # Connect to specific server
  python tcp_test_client.py --host 192.168.1.100 --port 2345

  # Custom timeout and output directory
  python tcp_test_client.py --timeout 60 --output ./my_images
      
  # Quick test against simulator
  python tcp_test_client.py --host localhost --port 12345 --timeout 30
        """
    )
    
    parser.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='TCP server host (default: localhost)'
    )
    
    parser.add_argument(
        '--port',
        type=int,
        default=12345,
        help='TCP server port (default: 12345)'
    )
    
    parser.add_argument(
        '--timeout',
        type=int,
        default=30,
        help='Connection and receive timeout in seconds (default: 30)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='./output',
        help='Output directory for saved images (default: ./output)'
    )
    
    args = parser.parse_args()
    
    client = TargetScreenClient(
        host=args.host,
        port=args.port,
        timeout=args.timeout,
        output_dir=args.output
    )
    
    return client.run()


if __name__ == '__main__':
    sys.exit(main())
