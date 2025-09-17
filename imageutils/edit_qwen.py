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
import threading
import queue
import base64
from io import BytesIO
from PIL import Image
from concurrent.futures import Future
from typing import List, Optional, Any


class QwenEditError(Exception):
    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception

class QwenEditClient:
    def __init__(
        self,
        server_urls: List[str],
        num_workers: Optional[int] = None,
        request_timeout: int = 120
    ):
        if not server_urls or not all(isinstance(url, str) for url in server_urls):
            raise ValueError("server_urls must be a non-empty list of strings.")
        
        self.server_urls = server_urls
        self.num_servers = len(server_urls)
        self.timeout = request_timeout
        
        if num_workers is None:
            num_workers = self.num_servers
        
        print(f"QwenEditClient initialized with {self.num_servers} server instances, {num_workers} internal workers.")

        self.task_queue = queue.Queue()
        self._workers = []
        self._active = True

        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"EditWorker-{i}")
            worker.daemon = True
            worker.start()
            self._workers.append(worker)

        self._lock = threading.Lock()
        self._server_index = 0
        self.session = requests.Session()

    def _get_next_server_url(self) -> str:
        with self._lock:
            url = self.server_urls[self._server_index]
            self._server_index = (self._server_index + 1) % self.num_servers
            return url

    def _worker_loop(self):
        while self._active:
            try:
                task = self.task_queue.get(timeout=1)
                if task is None:
                    break
                
                image, prompt, future, kwargs = task
                
                try:
                    result_image = self._send_request(image, prompt, **kwargs)
                    future.set_result(result_image)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                continue
    
    def _send_request(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_b64_string = base64.b64encode(buffered.getvalue()).decode("utf-8")

        server_url = self._get_next_server_url()
        endpoint = f"{server_url}/edit/"

        payload = {
            "image_b64": image_b64_string,
            "prompt": prompt,
            **kwargs
        }

        try:
            response = self.session.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("images"):
                img_b64 = data["images"][0]
                img_bytes = base64.b64decode(img_b64)
                return Image.open(BytesIO(img_bytes))
            else:
                raise QwenEditError(f"Fail to edit image: {data.get('detail', 'N/A')}")
        except requests.exceptions.RequestException as e:
            raise QwenEditError(f"Fail to edit image at {server_url}: {e}", original_exception=e)
        
    def edit_image(self, image: Image.Image, prompt: str, **kwargs: Any) -> Image.Image:
        if not self._active:
            raise RuntimeError("QwenEditClient is not active.")

        future = Future()
        self.task_queue.put((image, prompt, future, kwargs))
        
        return future.result()

    def shutdown(self, wait: bool = True):
        if not self._active:
            return
            
        print("Shutdown QwenEditClient...")
        self._active = False
        for _ in self._workers:
            self.task_queue.put(None)
        
        if wait:
            for worker in self._workers:
                worker.join()
        print("QwenEditClient shutdown completed.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()
