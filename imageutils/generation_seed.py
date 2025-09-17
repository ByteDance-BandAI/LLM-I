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


class SeedreamClient:
    def __init__(
        self,
        base_url: str = None,
        api_key: str = None,
        model_name: str = None,
        retry_interval: int = 30,
        retry_times: int = 3,
    ):
        self.api_key = api_key
        self.retry_interval = retry_interval
        self.retry_times = retry_times
        self.model_name = model_name

        if not self.api_key:
            self.api_key = os.environ.get("ARK_API_KEY")

        self.client = OpenAI(
            base_url=base_url,
            api_key=self.api_key,
        )

    def generate_image_once(self, prompt: str, format="url"):
        try:
            response = self.client.images.generate(
                model=self.model_name,
                prompt=prompt,
                response_format=format,
                extra_body={"watermark": False},
            )
            return response.data[0].url
        except Exception as e:
            return None

    def generate_image(self, prompt: str, format="url"):
        try:
            response = self.client.images.generate(
                model=self.model_name,
                prompt=prompt,
                response_format=format,
                extra_body={"watermark": False},
            )
            rsp_url = response.data[0].url
            return rsp_url
        except Exception as e:
            if type(e) == RateLimitError:
                for i in range(1, self.retry_times):
                    print(f"[WARNING] RateLimitError, prompt: {prompt}, retry times: {i}")
                    time.sleep(self.retry_interval)
                    rsp_url = self.generate_image_once(prompt, format)
                    if rsp_url:
                        return rsp_url
                return None
            elif type(e) == BadRequestError:
                print(f"[WARNING] BadRequestError prompt: {prompt}")
                return None
            else:
                print(f"[ERROR] Unexpected Error: {e}")
                return None

    def get_pil_image(self, prompt: str, format="url"):
        rsp_url = self.generate_image(prompt, format)
        if rsp_url:
            try:
                with requests.get(rsp_url, timeout=2, stream=True) as response:
                    response.raise_for_status()
                    image_buffer = BytesIO(response.content)
                    with Image.open(image_buffer) as img:
                        image_copy = img.copy()
                return image_copy
            except:
                return None
        return None