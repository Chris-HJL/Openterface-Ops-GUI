#!/usr/bin/env python3
"""
UI-Model Server
Standalone UI-Model server with OpenAI-compatible API
"""

import requests
import json
import os
import datetime
import re
import torch
import argparse
import atexit
import signal
import sys
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Qwen3VLForConditionalGeneration
from typing import Dict, Any, Optional
from flask import Flask, request, jsonify

# UI-Model global variables
UI_MODEL_PATH = "D:\\AI\\models\\Fara-7B"
# UI_MODEL_PATH = "D:\\AI\\models\\UI-Ins-7B"
ui_model = None
ui_processor = None
standby_mode = "cold"  # Default cold standby mode

app = Flask(__name__)


def parse_coordinates(raw_string: str) -> tuple[int, int]:
    """
    Parse coordinates from model response, supports multiple formats:
    - Single coordinate: [x,y]
    - Two coordinates: [x1,y1,x2,y2]
    - More coordinates: [x1,y1,x2,y2,x3,y3,...]
    When multiple coordinates are present, only the first coordinate is extracted
    """
    # Match all sequences of numbers within square brackets
    matches = re.findall(r'\[([^\]]+)\]', raw_string)
    
    for match in matches:
        # Split numbers and convert to integers
        numbers = [int(x.strip()) for x in match.split(',') if x.strip().isdigit()]
        
        # If there are at least 2 numbers, return the first two as the first coordinate point
        if len(numbers) >= 2:
            return numbers[0], numbers[1]
    
    return -1, -1





def load_ui_model(model_path: str = UI_MODEL_PATH):
    """
    Load UI-Model
    """
    global ui_model, ui_processor
    print("Loading UI-Model...")
    ui_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        # attn_implementation="flash_attention_2",
    ).eval()
    ui_processor = AutoProcessor.from_pretrained(model_path)
    print("UI-Model loaded successfully")


def run_ui_model_inference(image_path: str, instruction: str) -> tuple[int, int]:
    """
    Run UI-Model inference to get coordinates
    
    Args:
        image_path: Path to the image
        instruction: Instruction text
        
    Returns:
        Coordinate point (x, y)
    """
    global ui_model, ui_processor
    
    # Check if model is already loaded
    if ui_model is None or ui_processor is None:
        load_ui_model()
    
    # Load image
    image = Image.open(image_path).convert("RGB")
    
    # Build messages
    messages = [
        {
            "role":"system",
            "content": "Provide the coordinate of the element in the screenshot. The coordinate should be in the format of [x, y], enclosed in square brackets."
        },
        {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]
        }]
    
    # Process input
    prompt = ui_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = ui_processor(text=[prompt], images=[image], return_tensors="pt").to(ui_model.device)
    
    # Generate response
    generated_ids = ui_model.generate(
        **inputs,
        max_new_tokens=200,
        # temperature=0.0,
        # top_p=1.0,
        # top_k=-1,
    )
    response_ids = generated_ids[0, len(inputs["input_ids"][0]):]
    raw_response = ui_processor.decode(response_ids, skip_special_tokens=True)
    print(f"\nRaw model response: {raw_response}")
    
    # Parse coordinates
    point_x, point_y = parse_coordinates(raw_response)
    
    return point_x, point_y


def cleanup_model():
    """
    Clean up model resources when program exits
    """
    global ui_model, ui_processor
    if ui_model is not None or ui_processor is not None:
        print("Program exiting, cleaning up model resources...")
        if ui_model is not None:
            ui_model = None
        if ui_processor is not None:
            ui_processor = None
        # Force release GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("Model resources cleaned up successfully")


def unload_ui_model():
    """
    Unload UI-Model and release resources
    """
    global ui_model, ui_processor
    print("Unloading UI-Model...")
    if ui_model is not None:
        # Clear model and release GPU memory
        ui_model = None
    if ui_processor is not None:
        ui_processor = None
    # Force release GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("UI-Model has been unloaded")


def process_ui_element_request(image_path: str, instruction: str) -> Dict[str, Any]:
    """
    Process UI element localization request
    """
    print(f"\nLocating element: {instruction}")
    
    # Run UI-Model inference (will automatically load model if not loaded)
    point_x, point_y = run_ui_model_inference(image_path, instruction)
    
    if point_x != -1:
        print(f"Element located successfully at coordinates: ({point_x}, {point_y})")
        
        return {
            "success": True,
            "coordinates": [point_x, point_y],
            "message": f"Element found at coordinates ({point_x}, {point_y})"
        }
    else:
        print("Failed to parse coordinates")
        return {
            "success": False,
            "coordinates": None,
            "message": "Failed to parse coordinates"
        }

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    OpenAI-compatible chat completions API
    """
    try:
        data = request.json
        messages = data.get('messages', [])
        
        # Extract last user message
        user_message = None
        for msg in reversed(messages):
            if msg['role'] == 'user':
                user_message = msg
                break
        
        if not user_message:
            return jsonify({
                "error": {
                    "message": "No user message found",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }), 400
        
        # Extract instruction and image
        instruction = None
        image_data = None
        
        content = user_message.get('content', '')
        if isinstance(content, list):
            # Process multimodal content
            for item in content:
                if item['type'] == 'text':
                    instruction = item['text']
                elif item['type'] == 'image_url':
                    image_url = item['image_url']['url']
                    # Process base64 image data
                    if image_url.startswith('data:image/'):
                        import base64
                        import io
                        # Extract base64 data
                        base64_data = image_url.split(',')[1]
                        # Decode base64 data to bytes
                        image_bytes = base64.b64decode(base64_data)
                        # Create temporary image file
                        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                        temp_image_path = f"./temp_images/ui_model_temp_{timestamp}.jpg"
                        os.makedirs("./temp_images", exist_ok=True)
                        with open(temp_image_path, 'wb') as f:
                            f.write(image_bytes)
                        image_data = temp_image_path
        else:
            # Plain text content
            instruction = content
        
        if not instruction:
            return jsonify({
                "error": {
                    "message": "No instruction found",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }), 400
        
        if not image_data:
            return jsonify({
                "error": {
                    "message": "No image found",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": None
                }
            }), 400
        
        # Process UI element localization request
        result = process_ui_element_request(image_data, instruction)
        
        # Decide whether to unload model based on standby mode
        if standby_mode == "cold":
            # Cold standby mode: unload model after each request
            unload_ui_model()
        # Hot standby mode: keep model loaded
        
        # Generate response
        if result['success']:
            response_content = f"Element found at coordinates [{result['coordinates'][0]}, {result['coordinates'][1]}]"
        else:
            response_content = "Failed to locate element"
        
        return jsonify({
            "id": f"chatcmpl-{datetime.datetime.now().timestamp()}",
            "object": "chat.completion",
            "created": int(datetime.datetime.now().timestamp()),
            "model": "ui-model",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content
                    },
                    "finish_reason": "stop"
                }
            ],
            "usage": {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        # Decide whether to unload model based on standby mode
        if standby_mode == "cold":
            # Cold standby mode: unload model even when exception occurs
            unload_ui_model()
        # Hot standby mode: keep model loaded
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_server_error",
                "param": None,
                "code": None
            }
        }), 500

@app.route('/v1/models', methods=['GET'])
def get_models():
    """
    Get list of available models
    """
    return jsonify({
        "object": "list",
        "data": [
            {
                "id": "ui-model-7b",
                "object": "model",
                "created": int(datetime.datetime.now().timestamp()),
                "owned_by": "ui-model",
                "root": "ui-model-7b",
                "parent": None,
                "permission": []
            }
        ]
    })

@app.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint
    """
    return jsonify({
        "status": "ok",
        "message": "UI-Model Server is running",
        "model_loaded": ui_model is not None,
        "standby_mode": standby_mode
    })

def signal_handler(signum, frame):
    """Signal handler to ensure resource cleanup when program exits"""
    print(f"\nReceived signal {signum}, shutting down server...")
    cleanup_model()
    sys.exit(0)


def main():
    global standby_mode
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="UI-Model Server - Standalone UI-Model server")
    parser.add_argument(
        "--standby-mode", 
        choices=["cold", "hot"], 
        default="cold",
        help="Standby mode: cold(unload model after each request) or hot(preload model on startup)"
    )
    parser.add_argument(
        "--host", 
        default="0.0.0.0",
        help="Server host address (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port", 
        type=int, 
        default=2345,
        help="Server port (default: 2345)"
    )
    parser.add_argument(
        "--model-path", 
        default=UI_MODEL_PATH,
        help=f"UI-Model path (default: {UI_MODEL_PATH})"
    )
    
    args = parser.parse_args()
    standby_mode = args.standby_mode
    
    # Register cleanup function for program exit
    atexit.register(cleanup_model)
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Decide whether to preload model based on standby mode
    if standby_mode == "hot":
        print(f"Hot standby mode: Preloading model...")
        try:
            load_ui_model(args.model_path)
            print("Model preloaded successfully")
        except Exception as e:
            print(f"Failed to preload model: {e}")
            print("Server will start in cold standby mode")
            standby_mode = "cold"
    else:
        print("Cold standby mode: Model will be loaded on first request")
    
    print(f"Standby mode: {standby_mode}")
    
    # Start server
    print(f"\nUI-Model Server is running on http://{args.host}:{args.port}")
    print("Available endpoints:")
    print(f"  GET  http://{args.host}:{args.port}/health")
    print(f"  GET  http://{args.host}:{args.port}/v1/models")
    print(f"  POST http://{args.host}:{args.port}/v1/chat/completions")
    print(f"\nStandby mode: {standby_mode}")
    
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
