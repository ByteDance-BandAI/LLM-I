#!/usr/bin/env python3
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import requests
import base64
import time
import os
import argparse
import sys
from PIL import Image
from io import BytesIO
from typing import Optional


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Test Qwen Image API endpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test generation endpoint
  python client.py --mode generation --prompt "a beautiful sunset over mountains"

  # Test edit endpoint with image file
  python client.py --mode edit --prompt "make it black and white" --image_path input.jpg

  # Test with custom server settings
  python client.py --mode generation --host localhost --port 8000 --prompt "a cat"

  # Test health check
  python client.py --mode health
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--mode",
        choices=["generation", "edit", "health", "info"],
        required=True,
        help="Test mode: 'generation', 'edit', 'health', or 'info'"
    )
    
    # Server connection arguments
    parser.add_argument(
        "--host",
        type=str,
        default="[::]",
        help="Server host address (default: [::])"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port number (default: 8000)"
    )
    
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=0,
        help="GPU ID to test (used for port calculation: base_port + gpu_id, default: 0)"
    )
    
    # Generation/Edit arguments
    parser.add_argument(
        "--prompt",
        type=str,
        help="Text prompt for generation or edit (required for generation/edit modes)"
    )
    
    parser.add_argument(
        "--image_path",
        type=str,
        help="Path to input image for edit mode (required for edit mode)"
    )
    
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="bad anatomy, watermark, low quality",
        help="Negative prompt (default: 'bad anatomy, watermark, low quality')"
    )
    
    parser.add_argument(
        "--height",
        type=int,
        default=1024,
        help="Image height in pixels (default: 1024)"
    )
    
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Image width in pixels (default: 1024)"
    )
    
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=25,
        help="Number of inference steps (default: 25 for generation, 50 for edit)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_images",
        help="Output directory for generated images (default: generated_images)"
    )
    
    parser.add_argument(
        "--output_filename",
        type=str,
        help="Output filename (auto-generated if not specified)"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=120,
        help="Request timeout in seconds (default: 120)"
    )
    
    return parser.parse_args()


def get_server_url(host: str, port: int, gpu_id: int) -> str:
    """Get the server URL based on host, port, and GPU ID."""
    actual_port = port + gpu_id
    return f"http://{host}:{actual_port}"


def test_health_check(host: str, port: int, gpu_id: int, timeout: int):
    """Test the health check endpoint."""
    url = get_server_url(host, port, gpu_id)
    endpoint = f"{url}/health"
    
    print(f"Testing health check at: {endpoint}")
    
    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Health check successful!")
        print(f"Response: {data}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check failed: {e}")
        return False


def test_info(host: str, port: int, gpu_id: int, timeout: int):
    """Test the info endpoint."""
    url = get_server_url(host, port, gpu_id)
    endpoint = f"{url}/info"
    
    print(f"Testing info endpoint at: {endpoint}")
    
    try:
        response = requests.get(endpoint, timeout=timeout)
        response.raise_for_status()
        
        data = response.json()
        print(f"✅ Info endpoint successful!")
        print(f"Response: {data}")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Info endpoint failed: {e}")
        return False


def test_generation(host: str, port: int, gpu_id: int, prompt: str, negative_prompt: str,
                   height: int, width: int, num_inference_steps: int, output_dir: str,
                   output_filename: Optional[str], timeout: int):
    """Test the image generation endpoint."""
    url = get_server_url(host, port, gpu_id)
    endpoint = f"{url}/generate/"
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_inference_steps": num_inference_steps,
    }
    
    print(f"Testing generation at: {endpoint}")
    print(f"Prompt: '{prompt}'")
    print(f"Parameters: {height}x{width}, {num_inference_steps} steps")
    
    start_time = time.time()
    
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        
        end_time = time.time()
        print(f"✅ Generation successful! Time taken: {end_time - start_time:.2f} seconds.")
        
        data = response.json()
        
        if data.get("status") == "success" and data.get("images"):
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename if not provided
            if not output_filename:
                timestamp = int(time.time())
                output_filename = f"generated_{timestamp}.png"
            
            # Ensure output filename has .png extension
            if not output_filename.endswith('.png'):
                output_filename += '.png'
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Decode and save the first image
            img_b64 = data["images"][0]
            img_bytes = base64.b64decode(img_b64)
            image = Image.open(BytesIO(img_bytes))
            
            image.save(output_path)
            print(f"✅ Image saved as '{output_path}'")
            return True
        else:
            print("❌ API call succeeded but did not return a valid image.")
            print(f"Response: {data}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Generation failed: {e}")
        return False


def test_edit(host: str, port: int, gpu_id: int, image_path: str, prompt: str,
              negative_prompt: Optional[str], num_inference_steps: int, output_dir: str,
              output_filename: Optional[str], timeout: int):
    """Test the image edit endpoint."""
    url = get_server_url(host, port, gpu_id)
    endpoint = f"{url}/edit/"
    
    # Load and encode the input image
    try:
        with open(image_path, "rb") as img_file:
            img_bytes = img_file.read()
            img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    except FileNotFoundError:
        print(f"❌ Input image file not found: {image_path}")
        return False
    except Exception as e:
        print(f"❌ Error loading input image: {e}")
        return False
    
    payload = {
        "image_b64": img_b64,
        "prompt": prompt,
        "num_inference_steps": num_inference_steps,
    }
    
    if negative_prompt:
        payload["negative_prompt"] = negative_prompt
    
    print(f"Testing edit at: {endpoint}")
    print(f"Input image: {image_path}")
    print(f"Prompt: '{prompt}'")
    print(f"Parameters: {num_inference_steps} steps")
    
    start_time = time.time()
    
    try:
        response = requests.post(endpoint, json=payload, timeout=timeout)
        response.raise_for_status()
        
        end_time = time.time()
        print(f"✅ Edit successful! Time taken: {end_time - start_time:.2f} seconds.")
        
        data = response.json()
        
        if data.get("status") == "success" and data.get("images"):
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename if not provided
            if not output_filename:
                timestamp = int(time.time())
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_filename = f"edited_{base_name}_{timestamp}.png"
            
            # Ensure output filename has .png extension
            if not output_filename.endswith('.png'):
                output_filename += '.png'
            
            output_path = os.path.join(output_dir, output_filename)
            
            # Decode and save the first image
            img_b64 = data["images"][0]
            img_bytes = base64.b64decode(img_b64)
            image = Image.open(BytesIO(img_bytes))
            
            image.save(output_path)
            print(f"✅ Edited image saved as '{output_path}'")
            return True
        else:
            print("❌ API call succeeded but did not return a valid image.")
            print(f"Response: {data}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"❌ Edit failed: {e}")
        return False


def main():
    """Main entry point."""
    args = parse_arguments()
    
    # Validate arguments based on mode
    if args.mode in ["generation", "edit"] and not args.prompt:
        print("❌ Error: --prompt is required for generation and edit modes")
        sys.exit(1)
    
    if args.mode == "edit" and not args.image_path:
        print("❌ Error: --image_path is required for edit mode")
        sys.exit(1)
    
    print(f"Testing Qwen Image API")
    print(f"Server: {args.host}:{args.port + args.gpu_id}")
    print(f"Mode: {args.mode}")
    print("-" * 50)
    
    # Test based on mode
    if args.mode == "health":
        success = test_health_check(args.host, args.port, args.gpu_id, args.timeout)
    elif args.mode == "info":
        success = test_info(args.host, args.port, args.gpu_id, args.timeout)
    elif args.mode == "generation":
        success = test_generation(
            args.host, args.port, args.gpu_id, args.prompt, args.negative_prompt,
            args.height, args.width, args.num_inference_steps, args.output_dir,
            args.output_filename, args.timeout
        )
    elif args.mode == "edit":
        success = test_edit(
            args.host, args.port, args.gpu_id, args.image_path, args.prompt,
            args.negative_prompt, args.num_inference_steps, args.output_dir,
            args.output_filename, args.timeout
        )
    
    if success:
        print("\n🎉 Test completed successfully!")
        sys.exit(0)
    else:
        print("\n💥 Test failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()