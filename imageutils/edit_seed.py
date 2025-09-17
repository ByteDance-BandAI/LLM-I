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
from openai import OpenAI
from openai import BadRequestError, RateLimitError
import requests
from PIL import Image
from io import BytesIO
import time
from volcenginesdkarkruntime import Ark
import base64
from typing import Any


class SeedEditClient:
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model_name: str = None,
        retry_interval: int = 20,
        retry_times: int = 3,
    ):
        self.api_key = api_key
        self.retry_interval = retry_interval
        self.retry_times = retry_times

        if not self.api_key:
            self.api_key = os.environ.get("ARK_API_KEY")

        self.client = Ark(
            base_url=base_url,
            api_key=self.api_key,
        )
        self.model = model_name
    
    def pil2base64(self, image: Image) -> str:
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"
    
    def download_image(self, url: str) -> Image.Image:
        try:
            with requests.get(url, timeout=2, stream=True) as response:
                response.raise_for_status()
                image_buffer = BytesIO(response.content)
                with Image.open(image_buffer) as img:
                    image_copy = img.copy()
                return image_copy
        except Exception as e:
            print(f"[ERROR] Download image failed: {e}")
            return None

    def edit_image(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        base64_image = self.pil2base64(image)
        for i in range(self.retry_times):
            url = self.edit_image_once(base64_image, prompt, **kwargs)
            if url:
                pil_img = self.download_image(url)
                if pil_img:
                    return pil_img
            time.sleep(self.retry_interval)
        return None
    
    def edit_image_once(self, base64_image: str, prompt: str, **kwargs: Any) -> str:
        try:
            response = self.client.images.generate(
                model=self.model,
                image=base64_image,
                prompt=prompt,
                watermark=False,
                **kwargs
            )
            return response.data[0].url
        except Exception as e:
            print(f"[ERROR] SeedEditClient edit image failed: {e}")
            return None