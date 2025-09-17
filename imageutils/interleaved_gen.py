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


import re
import os
import requests

from PIL.Image import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

from imageutils.code_exec import codeexec
from imageutils.generation_seed import SeedreamClient
from imageutils.search_client import SearchClient
from imageutils.generation_qwen import QwenImageClient
from imageutils.edit_qwen import QwenEditClient
from imageutils.edit_seed import SeedEditClient
from verl.utils.reward_score.llmi_rm import is_image_valid, parse_image_tag


class InterleavedGenerator:
    def __init__(
        self,
        diffusion_backbone: str = "seed",
        edit_backbone: str = "seed",
        search_topk: int = 15,
    ):
        if diffusion_backbone == "seed":
            self.diffusion_client = SeedreamClient()
        elif diffusion_backbone == "qwen":
            self.diffusion_client = QwenImageClient(None)
        else:
            raise ValueError(f"Not Implemented diffusion client: {diffusion_backbone}.")

        if edit_backbone == "seed":
            self.edit_client = SeedEditClient()
        elif edit_backbone == "qwen":
            self.edit_client = QwenEditClient(None)
        else:
            raise ValueError(f"Not Implemented edit client: {edit_backbone}.")

        self.search_client = SearchClient(topk=search_topk)

    def tag2image(
        self, imagetag: str, tag_len: int, multimodal_inputs: list[Image] = None
    ) -> tuple[Image | None, str | None]:
        if not multimodal_inputs:
            mm_len = 0
        else:
            mm_len = len(multimodal_inputs)

        if not is_image_valid(imagetag, tag_len, mm_len):
            return None, None

        data = parse_image_tag(imagetag)
        source = data["source"]

        if source == "diffusion":
            prompt = data["params"]["prompt"]
            image = self.diffusion_client.get_pil_image(prompt)
        elif source == "search":
            prompt = data["params"]["query"]
            image = self.search_client.get_pil_image(prompt)
        elif source == "code":
            code = data["params"]["code"]
            image, _ = codeexec(code)
        else:  # edit
            if not multimodal_inputs:
                image = None
            else:
                prompt = data["params"]["prompt"]
                img_num = data["params"]["img_index"]
                try:
                    img = multimodal_inputs[int(img_num)].copy()
                    image = self.edit_client.edit_image(img, prompt)
                except:
                    image = None
        return image, source

    def interleaved_generation_concurrent(
        self, content: str, multimodal_inputs: list[Image] = None, max_workers: int = 8
    ) -> tuple[str, list[Image], dict]:
        pattern = r"<imgen>({.*?})</imgen>"
        image_tags_info = re.findall(pattern, content, flags=re.DOTALL)

        if not image_tags_info:
            empty_stats = {
                "diffusion": {"total": 0, "success": 0, "success_rate": 1.0},
                "search": {"total": 0, "success": 0, "success_rate": 1.0},
                "code": {"total": 0, "success": 0, "success_rate": 1.0},
                "edit": {"total": 0, "success": 0, "success_rate": 1.0},
            }
            return content, [], empty_stats

        initial_mm_len = len(multimodal_inputs) if multimodal_inputs else 0
        total_tags = len(image_tags_info)

        jobs = []
        for i, tag_info in enumerate(image_tags_info):
            data = parse_image_tag(tag_info)

            if not isinstance(data, dict) or "source" not in data:
                job = {
                    "tag_info": tag_info,
                    "source": None,
                    "params": {},
                    "original_index": i,
                    "dependency": float("inf"),
                }
                jobs.append(job)
                continue

            job = {
                "tag_info": tag_info,
                "source": data["source"],
                "params": data.get("params", {}),
                "original_index": i,
                "dependency": None,
            }

            if data["source"] == "edit":
                try:
                    img_index = int(data["params"].get("img_index", -1))
                    total_possible_images = initial_mm_len + total_tags

                    if not (0 <= img_index < total_possible_images):
                        job["dependency"] = float("inf")
                    elif img_index >= initial_mm_len:
                        dependency_on_job_index = img_index - initial_mm_len
                        if dependency_on_job_index >= i:
                            job["dependency"] = float("inf")
                        else:
                            job["dependency"] = dependency_on_job_index
                except Exception as e:
                    print(f"Error parsing img_index for edit job: {e}")
                    job["dependency"] = float("inf")

            jobs.append(job)

        results = [None] * len(jobs)
        pending_jobs_indices = set(range(len(jobs)))

        while pending_jobs_indices:
            current_new_images = [
                res[0] if res is not None else None for res in results
            ]
            live_image_pool = (
                multimodal_inputs if multimodal_inputs else []
            ) + current_new_images

            runnable_batch_indices = []
            for idx in pending_jobs_indices:
                dep = jobs[idx]["dependency"]
                is_runnable = False
                if dep is None:
                    is_runnable = True
                elif isinstance(dep, int):
                    if results[dep] is not None:
                        is_runnable = True
                if is_runnable:
                    runnable_batch_indices.append(idx)

            if not runnable_batch_indices:
                for idx in pending_jobs_indices:
                    results[idx] = (None, jobs[idx]["source"])
                break

            batch_results = {}
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_index = {
                    executor.submit(
                        self.tag2image,
                        jobs[idx]["tag_info"],
                        total_tags,
                        live_image_pool,
                    ): idx
                    for idx in runnable_batch_indices
                }
                for future in as_completed(future_to_index):
                    original_index = future_to_index[future]
                    try:
                        batch_results[original_index] = future.result()
                    except Exception as exc:
                        print(f"Job {original_index} generated an exception: {exc}")
                        batch_results[original_index] = (
                            None,
                            jobs[original_index]["source"],
                        )

            for index, result_tuple in batch_results.items():
                results[index] = result_tuple
                pending_jobs_indices.remove(index)

        stats = {
            "diffusion": {"total": 0, "success": 0},
            "search": {"total": 0, "success": 0},
            "code": {"total": 0, "success": 0},
            "edit": {"total": 0, "success": 0},
        }
        pil_images_list = []
        for image, source in results:
            if source is None:
                continue
            stats[source]["total"] += 1
            if isinstance(image, Image):
                pil_images_list.append(image)
                stats[source]["success"] += 1

        results_iterator = iter(results)

        def replacement_logic(match: re.Match) -> str:
            image, _ = next(results_iterator)
            return "<image>" if isinstance(image, Image) else "<fail_to_generate_image>"

        processed_text = re.sub(pattern, replacement_logic, content, flags=re.DOTALL)

        stats_with_rate = {}
        for source, data in stats.items():
            total, success = data["total"], data["success"]
            stats_with_rate[source] = {
                "total": total,
                "success": success,
                "success_rate": round(success / total, 4) if total > 0 else 1.0,
            }

        return processed_text, pil_images_list, stats_with_rate
