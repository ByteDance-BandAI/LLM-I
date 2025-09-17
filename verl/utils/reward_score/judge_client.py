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

import random
import requests
import base64
from io import BytesIO
from PIL import Image
from typing import List, Dict, Any
from textwrap import dedent
from openai import OpenAI, DefaultHttpxClient
import os
import re


class MLLMJudgeClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        max_pixels: int = 28 * 28 * 1000,
        sp_path: str = "system_prompts/training/mllmjudge.txt",
        api_key: str = "EMPTY",
    ):
        if not base_url:
            return

        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultHttpxClient(trust_env=False, timeout=150),
        )

        self.max_pixels = max_pixels

        if model_name:
            self.model_name = model_name
        else:
            # Attempt to retrieve the model name
            sess = requests.Session()
            sess.trust_env = False
            resp = sess.get(f"{base_url}/models")
            resp = resp.json()

            if len(resp.get("data", [])) == 1:
                self.model_name = resp["data"][0]["id"]
            else:
                raise ValueError(
                    "Multiple or no models found. Please specify model_name."
                )

        print(
            f"MLLMJudgeClient initialized successfully, base_url={base_url}, model_name={self.model_name}"
        )

        self._system_prompt(sp_path)

        print(f"System Prompts of MLLMJudge: {self.system_prompt}")

    def _process_image(self, image: Image.Image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")

    def _system_prompt(self, sp_path: str):
        with open(sp_path, "r") as f:
            self.system_prompt = f.read()

    def _encode_image(self, image: Image.Image) -> str:
        image = self._process_image(image)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"

    def _build_messages(
        self, processed_text: str, pil_images_list: List[Image.Image]
    ) -> List[Dict[str, Any]]:

        text_segments = processed_text.split("<image>")

        message_content = []
        for i, text_segment in enumerate(text_segments):
            if text_segment:
                message_content.append({"type": "text", "text": text_segment})

            if i < len(pil_images_list):
                base64_image = self._encode_image(pil_images_list[i])
                message_content.append(
                    {"type": "image_url", "image_url": {"url": base64_image, "max_pixels": self.max_pixels}}
                )

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": message_content},
        ]

    def extract_score(self, answer: str):
        img_quality_score = re.search(r"Image Quality:\s*(-?\d+)", answer)
        ti_rel_score = re.search(r"Task Relevance:\s*(-?\d+)", answer)
        tq_rel_score = re.search(r"Text-Image Relevance:\s*(-?\d+)", answer)
        return img_quality_score, ti_rel_score, tq_rel_score

    def _judge_answer(
        self, processed_text: str, pil_images_list: List[Image.Image], **kwargs
    ) -> str:
        if not self.client:
            raise ConnectionError(
                "Client is not initialized. Please provide a valid base_url."
            )

        messages = self._build_messages(processed_text, pil_images_list)

        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, **kwargs
        )

        return response.choices[0].message.content.strip()

    def get_test_prompt(self, answer: str, question: str):
        test_prompt = f"Question: {question}\nAnswer: {answer}"
        return test_prompt

    def verify(
        self, answer: str, question: str, pil_images_list: List[Image.Image]
    ) -> tuple[int, int]:

        if not pil_images_list:
            return 5, 5, 5

        test_prompt = self.get_test_prompt(answer, question)
        try:
            response = self._judge_answer(test_prompt, pil_images_list)
        except Exception as e:
            print(f"MLLM Judge error: {e}")
            return 1, 1, 1
        img_quality_score, ti_rel_score, tq_rel_score = self.extract_score(response)
        img_quality_score = int(img_quality_score.group(1)) if img_quality_score else 1
        ti_rel_score = int(ti_rel_score.group(1)) if ti_rel_score else 1
        tq_rel_score = int(tq_rel_score.group(1)) if tq_rel_score else 1

        return img_quality_score, ti_rel_score, tq_rel_score


class LLMJudgeClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        sp_path: str = "system_prompts/training/llmjudge.txt",
        api_key: str = "EMPTY",
    ):
        if not base_url:
            return
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            http_client=DefaultHttpxClient(trust_env=False, timeout=100),
        )

        if model_name:
            self.model_name = model_name
        else:
            # Attempt to retrieve the model name
            sess = requests.Session()
            sess.trust_env = False
            resp = sess.get(f"{base_url}/models")
            resp = resp.json()
            assert (
                len(resp["data"]) == 1
            ), f"Multiple models are deployed on the current base_url, please specify model_name."
            self.model_name = resp["data"][0]["id"]
        print(
            f"LLMJudgeClient initialized successfully, base_url={base_url}, model_name={self.model_name}"
        )

        self._system_prompt(sp_path)
        print(f"System Prompt of LLMJudge: {self.system_prompt}")

    def get_test_prompt(self, answer: str, question: str):
        test_prompt = f"Question: {question}\nAnswer: {answer}"
        return test_prompt

    def _system_prompt(self, sp_path: str):
        with open(sp_path, "r") as f:
            self.system_prompt = f.read()

    def extract_score(self, output: str):
        clean = re.sub(r"<think>.*?</think>", "", output, flags=re.DOTALL)
        text_m = re.search(r"Text:\s*(-?\d+)", clean)
        image_m = re.search(r"Image:\s*(-?\d+)", clean)
        return text_m, image_m

    def verify(self, answer: str, question: str):
        if '<think>' in answer:
            print('[DEBUG] LLMJudge: think still in response str')
        test_prompt = self.get_test_prompt(answer, question)
        try:
            chat_response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": test_prompt},
                ],
                seed=random.randint(0, 1000000),
                temperature=0.5,
                extra_body={
                    "enable_thinking": False
                }
            )
            response = chat_response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM Judge error: {e}")
            return 1, 1
        text_m, image_m = self.extract_score(response)
        text_score = int(text_m.group(1)) if text_m else 1
        image_score = int(image_m.group(1)) if image_m else 1

        return text_score, image_score