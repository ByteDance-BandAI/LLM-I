# Qwen Image Deployment

This directory contains a unified deployment solution for both Qwen-Image generation and Qwen-Image-Edit models with multi-GPU support.

## Files

- `launcher.py`: Unified launcher script with command-line argument parsing
- `server.py`: FastAPI server that supports both generation and edit modes
- `client.py`: Test client for validating server functionality

## Usage

### Basic Usage

```bash
# Launch generation model
python launcher.py --mode generation --model_path /path/to/qwen-image --gpus 0,1

# Launch edit model  
python launcher.py --mode edit --model_path /path/to/qwen-image-edit --gpus 0
```

### Advanced Usage

```bash
# Custom host, port, and workers
python launcher.py \
    --mode generation \
    --model_path /path/to/qwen-image \
    --gpus 0,1,2 \
    --host 0.0.0.0 \
    --base_port 8000 \
    --workers_per_gpu 2 \
    --log_level info
```

### Command Line Arguments

- `--mode`: Model mode - either "generation" or "edit" (required)
- `--model_path`: Path to the model directory (required)
- `--gpus`: Comma-separated list of GPU IDs (required)
- `--host`: Host address to bind to (default: "::")
- `--base_port`: Base port number (default: 8000)
- `--workers_per_gpu`: Number of workers per GPU (default: 1)
- `--startup_delay`: Delay between starting processes (default: 1.0)
- `--log_level`: Log level (default: info)

### API Endpoints

#### Generation Mode
- `POST /generate/`: Generate images from text prompts
- `GET /health`: Health check
- `GET /info`: Get model information

#### Edit Mode
- `POST /edit/`: Edit images based on text instructions
- `GET /health`: Health check  
- `GET /info`: Get model information

### Multi-GPU Deployment

Each GPU will run on a separate port (base_port + gpu_id). For example:
- GPU 0: port 8000
- GPU 1: port 8001
- GPU 2: port 8002

### Testing the Server

Use the `client.py` script to test if your server is working correctly:

```bash
# Test health check
python client.py --mode health

# Test server info
python client.py --mode info

# Test generation
python client.py --mode generation --prompt "a beautiful sunset over mountains"

# Test edit with image file
python client.py --mode edit --prompt "make it black and white" --image_path input.jpg

# Test with custom server settings
python client.py --mode generation --host localhost --port 8000 --gpu_id 1 --prompt "a cat"
```

#### Client Arguments

- `--mode`: Test mode - "health", "info", "generation", or "edit" (required)
- `--host`: Server host (default: localhost)
- `--port`: Server base port (default: 8000)
- `--gpu_id`: GPU ID to test (default: 0)
- `--prompt`: Text prompt (required for generation/edit)
- `--image_path`: Input image path (required for edit)
- `--negative_prompt`: Negative prompt (optional)
- `--height`, `--width`: Image dimensions (default: 1024x1024)
- `--num_inference_steps`: Inference steps (default: 25)
- `--output_dir`: Output directory (default: generated_images)
- `--output_filename`: Custom output filename (optional)
- `--timeout`: Request timeout in seconds (default: 120)

### Environment Variables

The launcher sets these environment variables for each process:
- `CUDA_VISIBLE_DEVICES`: GPU ID for the process
- `MODEL_PATH`: Path to the model
- `MODEL_MODE`: Either "generation" or "edit"

## Dependencies

- fastapi
- uvicorn
- torch
- diffusers
- pillow
- pydantic
