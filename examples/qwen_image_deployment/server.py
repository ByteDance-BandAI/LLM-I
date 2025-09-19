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
import torch
import base64
from io import BytesIO
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from diffusers import DiffusionPipeline, QwenImageEditPipeline
from PIL import Image
from typing import Optional


# --- 1. Initialize FastAPI application ---
app = FastAPI(
    title="Qwen-Image API",
    description="An API for generating and editing images using Qwen-Image models.",
    version="1.0.0"
)

# --- 2. Define request body models ---
class ImageRequest(BaseModel):
    prompt: str = Field(..., description="The main text prompt to generate the image.", example="a cat wearing a chef's hat and cooking")
    negative_prompt: str = Field(None, description="The negative prompt to guide the generation away from certain concepts.", example="blurry, low quality, ugly")
    height: int = Field(1024, description="The height of the generated image in pixels.", gt=0)
    width: int = Field(1024, description="The width of the generated image in pixels.", gt=0)
    num_inference_steps: int = Field(25, description="The number of denoising steps.", gt=0)
    guidance_scale: float = Field(7.5, description="Guidance scale for classifier-free guidance.", ge=0)
    num_images_per_prompt: int = Field(1, description="Number of images to generate per prompt.", ge=1, le=4)


class ImageEditRequest(BaseModel):
    image_b64: str = Field(..., description="Base64 encoded input image string.")
    prompt: str = Field(..., description="The instruction prompt for editing the image.", example="make the cat wear a party hat")
    negative_prompt: Optional[str] = Field(None, description="The negative prompt.", example="blurry, low quality")
    num_inference_steps: int = Field(50, description="The number of denoising steps.", gt=0)


# --- 3. Load model ---
# This function will be called when FastAPI starts up
@app.on_event("startup")
async def load_model():
    global pipeline, model_mode
    
    # Get model mode and path from environment variables
    model_mode = os.environ.get("MODEL_MODE", "generation")
    model_path = os.environ.get("MODEL_PATH", "Qwen/Qwen-Image")
    
    # Get GPU ID from environment variable, this is key for multi-GPU deployment
    # gpu_id = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    # device = f"cuda:{gpu_id}"
    # print(f"Loading model onto device: {device}")
    device = "cuda:0"
    
    # For clearer logs, we can print the physical GPU ID that the process thinks is visible
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "N/A")
    print(f"Process started. Physical GPU requested: {visible_devices}. Loading model onto logical device: {device}")
    print(f"Model mode: {model_mode}")
    print(f"Model path: {model_path}")

    # Use float16 to save memory and speed up, 96GB memory is more than enough
    torch_dtype = torch.bfloat16

    try:
        if model_mode == "edit":
            pipeline = QwenImageEditPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
        else:
            pipeline = DiffusionPipeline.from_pretrained(model_path, torch_dtype=torch_dtype)
        
        # Move model to specified GPU
        pipeline.to(device)
        print(f"Model loaded successfully on {device} in {model_mode} mode.")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise RuntimeError("Failed to load the model.") from e


# --- 4. Define API endpoints ---
@app.post("/generate/", response_model=dict)
async def generate_image(request: ImageRequest):
    """
    Generates an image based on the provided prompt and parameters.
    Returns a JSON object with a list of base64-encoded images.
    """
    if model_mode != "generation":
        raise HTTPException(status_code=400, detail="This endpoint is only available in generation mode")
    
    try:
        print(f"Received generation request with prompt: {request.prompt}")

        # Call pipeline to generate images
        images = pipeline(
            prompt=request.prompt,
            negative_prompt=request.negative_prompt,
            height=request.height,
            width=request.width,
            num_inference_steps=request.num_inference_steps
        ).images
        print(len(images))
        # Convert generated PIL Image objects to Base64 strings
        encoded_images = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            encoded_images.append(img_str)

        print(f"Successfully generated {len(encoded_images)} image(s).")
        return {"status": "success", "images": encoded_images}

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/edit/", response_model=dict)
async def edit_image(request: ImageEditRequest):
    """
    Edits an image based on the provided prompt and parameters.
    Returns a JSON object with a list of base64-encoded images.
    """
    if model_mode != "edit":
        raise HTTPException(status_code=400, detail="This endpoint is only available in edit mode")
    
    try:
        if not request.image_b64:
            print("ERROR: Received empty image_b64 string.")
            raise ValueError("Input image_b64 string is empty.")

        # 2. Decode Base64 image
        image_bytes = base64.b64decode(request.image_b64)
        print(f"Decoded to {len(image_bytes)} bytes.")

        # 3. Try to open image
        input_image = Image.open(BytesIO(image_bytes)).convert("RGB")
        print(f"Successfully opened image. Format: {input_image.format}, Size: {input_image.size}")

        pipeline_args = {
            "prompt": request.prompt,
            "image": input_image,
            "num_inference_steps": request.num_inference_steps
        }
        if request.negative_prompt:
            pipeline_args["negative_prompt"] = request.negative_prompt

        images = pipeline(
            **pipeline_args
        ).images
        # Convert generated PIL Image objects to Base64 strings
        encoded_images = []
        for img in images:
            buffered = BytesIO()
            img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
            encoded_images.append(img_str)

        print(f"Successfully generated {len(encoded_images)} image(s).")
        return {"status": "success", "images": encoded_images}

    except Exception as e:
        print(f"An error occurred during image generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# --- 5. Health check endpoint ---
@app.get("/health")
def health_check():
    return {"status": "ok", "mode": model_mode}


@app.get("/info")
def get_info():
    """Get information about the current model and configuration."""
    return {
        "mode": model_mode,
        "model_path": os.environ.get("MODEL_PATH", "N/A"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "N/A"),
        "available_endpoints": ["/generate/"] if model_mode == "generation" else ["/edit/"]
    }