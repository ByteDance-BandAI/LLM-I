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
from PIL import Image, UnidentifiedImageError
import io
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional
import os


class SearchClient:

    def __init__(
        self,
        api_key: str = None,
        url: str = "https://serpapi.com/search",
        topk: int = 15,
        engine: str = "google_images",
        timeout: int = 2,
        user_agent: str = "Mozilla/5.0",
        retry_times: int = 3,
    ):
        self.api_key = api_key
        if not self.api_key:
            self.api_key = os.environ.get("SERP_API_KEY")
        self.url = url
        self.topk = topk
        self.engine = engine
        self.timeout = timeout
        self.user_agent = user_agent
        self.MIN_RESOLUTION = 400 * 400
        self.MAX_RESOLUTION = 1024 * 1024 * 5 * 5
        self.retry_times = retry_times
        self._validate_connection()
    
    def search_multimedia(self, query: str) -> List[str]:
        search_params = {
            "q": query,
            "engine": self.engine,
            "api_key": self.api_key,
        }
        response = requests.get(self.url, params=search_params).json()

        if response.ok() and "error" not in response.json():
            result = response.json().get('images_results', [])[:self.topk]
        else:
            result = None
        return result

    def _validate_connection(self) -> None:
        res = self.search_multimedia(
            query="a cat with a hat",
        )
        if res:
            print(
                f"Search Client Init Success, topk={self.topk}"
            )

    def _search(self, prompt: str, try_time: int) -> List[str]:
        res = self.search_multimedia(
            query=prompt,
        )
        if res:
            urls = []
            for item in res.result:
                urls.append(item["url"])
            return urls
        else:
            print(f"Search Error: {res.status}, try_time: {try_time}")
            return []

    def _convert_to_image(self, image_bytes: bytes) -> Image.Image | None:
        image_buffer = io.BytesIO(image_bytes)
        with Image.open(image_buffer) as img:
            img.verify()
        image_buffer.seek(0)
        pil_image = Image.open(image_buffer)
        img_copy = pil_image.copy()
        pil_image.close()
        return img_copy

    def _download_image(self, url: str) -> Image.Image | None:
        try:
            headers = {"User-Agent": self.user_agent}
            response = requests.get(url, timeout=self.timeout, headers=headers)
            response.raise_for_status()
            pil_image = self._convert_to_image(response.content)

            width, height = pil_image.size
            resolution = width * height
            if self.MIN_RESOLUTION <= resolution <= self.MAX_RESOLUTION:
                return pil_image
            return None

        except requests.exceptions.RequestException as e:
            pass
        except UnidentifiedImageError:
            pass
        except Exception as e:
            print(f"[ERROR] {e}")
        return None

    def get_pil_image(self, prompt: str) -> Optional[Image.Image]:
        try:
            for t in range(self.retry_times):
                url_list = self._search(prompt, t)
                if url_list:
                    break
        except:
            return None

        if not url_list:
            print("Search Error: No URLs found.")
            return None
        with ThreadPoolExecutor(max_workers=self.topk) as executor:
            future_to_url = {
                executor.submit(self._download_image, url): url for url in url_list
            }
            futures_in_order = list(future_to_url.keys())

            for i, future in enumerate(futures_in_order):
                try:
                    pil_image = future.result()
                    if pil_image:
                        for f in futures_in_order[i + 1 :]:
                            f.cancel()
                        return pil_image
                except Exception as e:
                    print(f"[ERROR] {e}")

        print(f"[WARNING] All {len(url_list)} URLs failed. Search Result: None")
        return None
