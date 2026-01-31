"""
Image server client module
"""
import socket
import datetime
import os
from typing import Optional
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
