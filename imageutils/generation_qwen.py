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


class QwenImageGenerationError(Exception):
    def __init__(self, message, original_exception=None):
        super().__init__(message)
        self.original_exception = original_exception


class QwenImageClient:
    def __init__(
        self,
        server_urls: List[str],
        num_workers: Optional[int] = None,
        request_timeout: int = 120
    ):

        if not server_urls or not all(isinstance(url, str) for url in server_urls):
            raise ValueError("server_urls must be a url list")
        
        self.server_urls = server_urls
        self.num_servers = len(server_urls)
        self.timeout = request_timeout
        
        if num_workers is None:
            num_workers = self.num_servers
        
        print(f"Client initialized, connected to {self.num_servers} server instances, {num_workers} internal worker threads started.")

        self.task_queue = queue.Queue()
        self._workers = []
        self._active = True

        for i in range(num_workers):
            worker = threading.Thread(target=self._worker_loop, name=f"Worker-{i}")
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
                
                prompt, future, kwargs = task
                
                try:
                    result_image = self._send_request(prompt, **kwargs)
                    future.set_result(result_image)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self.task_queue.task_done()

            except queue.Empty:
                continue
    
    def _send_request(self, prompt: str, **kwargs: Any) -> Image.Image:
        server_url = self._get_next_server_url()
        endpoint = f"{server_url}/generate/"
        payload = {"prompt": prompt, **kwargs}

        try:
            response = self.session.post(endpoint, json=payload, timeout=self.timeout)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "success" and data.get("images"):
                img_b64 = data["images"][0]
                img_bytes = base64.b64decode(img_b64)
                return Image.open(BytesIO(img_bytes))
            else:
                raise QwenImageGenerationError(f"Fail to generate image: {data.get('detail', 'N/A')}")
        except requests.exceptions.RequestException as e:
            raise QwenImageGenerationError(f"Fail to request {server_url}", original_exception=e)
        
    def get_pil_image(self, prompt: str, **kwargs: Any) -> Image.Image:
        if not self._active:
            raise RuntimeError("Client is closed, cannot submit new tasks.")

        future = Future()
        self.task_queue.put((prompt, future, kwargs))
        
        return future.result()

    def shutdown(self, wait: bool = True):
        if not self._active:
            return
            
        print("Shutting down client...")
        self._active = False
        for _ in self._workers:
            self.task_queue.put(None)
        
        if wait:
            for worker in self._workers:
                worker.join()
        print("Client shutdown complete.")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()