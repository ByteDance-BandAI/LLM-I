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

import os
import sys
import argparse
import multiprocessing
import subprocess
import time
from typing import List, Optional


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Qwen Image models on multiple GPUs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Launch generation model on GPUs 0,1 with custom model path
  python launcher.py --mode generation --model_path /path/to/qwen-image --gpus 0,1

  # Launch edit model on GPU 0 with custom host and port
  python launcher.py --mode edit --model_path /path/to/qwen-image-edit --gpus 0 --host 0.0.0.0 --port 8000

  # Launch with custom workers per GPU
  python launcher.py --mode generation --gpus 0,1,2 --workers_per_gpu 2
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--mode",
        choices=["generation", "edit"],
        required=True,
        help="Model mode: 'generation' for Qwen-Image or 'edit' for Qwen-Image-Edit"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        default="Qwen/Qwen-Image",
        help="Path to the model directory"
    )
    
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated list of GPU IDs to use (e.g., '0,1,2')"
    )
    
    # Optional arguments
    parser.add_argument(
        "--host",
        type=str,
        default="::",
        help="Host address to bind to (default: '::')"
    )
    
    parser.add_argument(
        "--base_port",
        type=int,
        default=8000,
        help="Base port number (default: 8000). Each GPU will use base_port + gpu_id"
    )
    
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=1,
        help="Number of workers per GPU (default: 1)"
    )
    
    parser.add_argument(
        "--startup_delay",
        type=float,
        default=1.0,
        help="Delay between starting each GPU process in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--log_level",
        choices=["debug", "info", "warning", "error"],
        default="info",
        help="Log level for the server (default: info)"
    )
    
    return parser.parse_args()


def parse_gpu_list(gpu_string: str) -> List[int]:
    """Parse comma-separated GPU string into list of integers."""
    try:
        gpus = [int(gpu.strip()) for gpu in gpu_string.split(",") if gpu.strip()]
        if not gpus:
            raise ValueError("No valid GPU IDs provided")
        return gpus
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"Invalid GPU list '{gpu_string}': {e}")


def run_server(gpu_id: int, mode: str, model_path: str, host: str, base_port: int, 
               workers_per_gpu: int, log_level: str):
    """
    Start a Uvicorn server process for the specified GPU.
    
    Args:
        gpu_id: GPU ID to use
        mode: Model mode ('generation' or 'edit')
        model_path: Path to the model directory
        host: Host address to bind to
        base_port: Base port number
        workers_per_gpu: Number of workers per GPU
        log_level: Log level for the server
    """
    port = base_port + gpu_id
    
    # Set environment variables for the subprocess
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["MODEL_PATH"] = model_path
    env["MODEL_MODE"] = mode
    
    print(f"Starting {mode} server for GPU {gpu_id} on port {port}...")
    print(f"Model path: {model_path}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    command = [
        "uvicorn",
        "server:app",
        "--host", host,
        "--port", str(port),
        "--workers", str(workers_per_gpu),
        "--log-level", log_level.lower()
    ]
    
    try:
        process = subprocess.Popen(
            command, 
            env=env, 
            cwd=script_dir
        )
        return process
    except Exception as e:
        print(f"Failed to start server for GPU {gpu_id}: {e}")
        return None


def main():
    """Main entry point."""
    # Parse command line arguments
    args = parse_arguments()
    
    # Parse GPU list
    try:
        gpus = parse_gpu_list(args.gpus)
    except argparse.ArgumentTypeError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    # Validate model path
    if not os.path.exists(args.model_path):
        print(f"Error: Model path '{args.model_path}' does not exist")
        sys.exit(1)
    
    # Set multiprocessing start method
    multiprocessing.set_start_method("spawn", force=True)
    
    print(f"Launching {args.mode} model on GPUs: {gpus}")
    print(f"Model path: {args.model_path}")
    print(f"Host: {args.host}")
    print(f"Base port: {args.base_port}")
    print(f"Workers per GPU: {args.workers_per_gpu}")
    print("-" * 50)
    
    # Start server processes
    processes = []
    for gpu_id in gpus:
        process = run_server(
            gpu_id=gpu_id,
            mode=args.mode,
            model_path=args.model_path,
            host=args.host,
            base_port=args.base_port,
            workers_per_gpu=args.workers_per_gpu,
            log_level=args.log_level
        )
        
        if process:
            processes.append(process)
            # Add delay between starting processes
            time.sleep(args.startup_delay)
        else:
            print(f"Failed to start process for GPU {gpu_id}")
    
    if not processes:
        print("Error: No server processes were started successfully")
        sys.exit(1)
    
    print(f"Successfully launched {len(processes)} server processes.")
    print("Press Ctrl+C to stop all servers.")
    
    try:
        # Wait for all processes to complete
        for process in processes:
            process.wait()
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        for process in processes:
            process.terminate()
        print("All servers stopped.")


if __name__ == "__main__":
    main()