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

from verl import DataProto
import torch
from collections import defaultdict
import re
import os
import json
from verl.workers.reward_manager import register
from verl.utils.reward_score.llmi_rm import compute_score as default_compute_score
from imageutils.interleaved_gen import InterleavedGenerator

from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional
import time
from functools import partial


@register("llmi")
class LLMIRewardManager:
    """The reward manager."""

    def __init__(
        self,
        config,
        tokenizer,
        num_examine,
        compute_score=None,
        reward_fn_key="data_source",
        max_resp_len=None,
        overlong_buffer_cfg=None,
        tool_reward_cfg=None,
        vlm_max_workers=6,
        score_max_workers=8,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        
        # Get backbone configuration from config
        diffusion_backbone = getattr(config, 'diffusion_backbone', 'seed')
        edit_backbone = getattr(config, 'edit_backbone', 'seed')
        
        print(f"[LLMI Reward Manager] Using diffusion_backbone: {diffusion_backbone}")
        print(f"[LLMI Reward Manager] Using edit_backbone: {edit_backbone}")
        
        self.interleaved_generator = InterleavedGenerator(
            diffusion_backbone=diffusion_backbone,
            edit_backbone=edit_backbone
        )
        self.compute_score = default_compute_score
        self.reward_fn_key = reward_fn_key
        self.max_resp_len = max_resp_len
        self.vlm_max_workers = vlm_max_workers
        self.score_max_workers = score_max_workers

        self.llm_text_factor = config.llm_cfg.get("llm_text_factor", 1.0)
        self.llm_image_factor = config.llm_cfg.get("llm_image_factor", 1.0)
        self.mllm_quality_factor = config.mllm_cfg.get("mllm_quality_factor", 1.0)
        self.mllm_ti_factor = config.mllm_cfg.get("mllm_ti_factor", 1.0)
        self.mllm_tq_factor = config.mllm_cfg.get("mllm_tq_factor", 1.0)
        self.llm_factor = config.llm_cfg.get("llm_factor", 1.0)
        self.mllm_factor = config.mllm_cfg.get("mllm_factor", 1.0)
        self.img_num_factor = config.get("img_num_factor", 1.0)

    def _init_reward_extra_info(self, num_samples: int) -> Dict[str, List]:
        reward_keys = [
            "img_num_score", "llm_text_score", "llm_image_score", "img_quality_score",
            "ti_rel_score", "tq_rel_score", "llm_score", "mllm_score", "r_tool", "score",
            "stats/diffusion_total", "stats/diffusion_success", "stats/diffusion_success_rate",
            "stats/search_total", "stats/search_success", "stats/search_success_rate",
            "stats/code_total", "stats/code_success", "stats/code_success_rate",
            "stats/edit_total", "stats/edit_success", "stats/edit_success_rate"
        ]
        return {key: [None] * num_samples for key in reward_keys}

    def _submit_vlm_tasks(self, data: DataProto, executor: ThreadPoolExecutor) -> Dict:
        future_vlm_to_info = {}
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]

            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)

            image_num = data_item.non_tensor_batch["extra_info"]["image_num"]
            data_source = data_item.non_tensor_batch['data_source']
            extra_info = data_item.non_tensor_batch.get("extra_info", None)

            multimodal_input = data_item.non_tensor_batch.get("multi_modal_data", None)
            if multimodal_input:
                image_list = multimodal_input.get("image", None)
            else:
                image_list = None

            mm_input_len = len(image_list) if image_list else 0

            if '<think>' in response_str:
                response_str = re.sub(r"<think>.*?</think>", "", response_str, flags=re.DOTALL)

            intergen_partial = partial(self.interleaved_generator.interleaved_generation_concurrent, multimodal_inputs=image_list)

            future = executor.submit(intergen_partial, response_str)
            future_vlm_to_info[future] = {
                "index": i, "prompt_str": prompt_str, "response_str": response_str,
                "valid_response_length": valid_response_length, "image_num": image_num,
                "data_source": data_source, "extra_info": extra_info, "mm_input_len": mm_input_len
            }
        return future_vlm_to_info

    def _process_vlm_results(self, future_vlm_to_info: Dict, score_executor: ThreadPoolExecutor,
                            reward_tensor: torch.Tensor, reward_extra_info: Dict) -> Dict:
        future_score_to_info = {}
        for future_vlm in as_completed(future_vlm_to_info):
            info_vlm = future_vlm_to_info[future_vlm]
            i = info_vlm["index"]
            try:
                vlm_response_str, pil_images_list, stats = future_vlm.result()
                future_score = score_executor.submit(
                    self.compute_score, solution_str=info_vlm["response_str"], 
                    image_num=info_vlm["image_num"], data_source=info_vlm["data_source"],
                    vlm_response_str=vlm_response_str, pil_img_list=pil_images_list,
                    mm_input_len=info_vlm["mm_input_len"], tool_stats=stats, extra_info=info_vlm["extra_info"]
                )
                future_score_to_info[future_score] = {**info_vlm, "vlm_response_str": vlm_response_str, "stats": stats}
            except Exception as exc:
                self._handle_processing_error(i, exc, reward_tensor, reward_extra_info, "VLM", info_vlm["valid_response_length"])
        return future_score_to_info

    def _process_score_results(self, future_score_to_info: Dict, reward_tensor: torch.Tensor, 
                              reward_extra_info: Dict, already_print_data_sources: Dict):
        for future_score in as_completed(future_score_to_info):
            info = future_score_to_info[future_score]
            i = info["index"]
            data_source = info["data_source"]
            valid_response_length = info["valid_response_length"]
            stats = info["stats"]

            try:
                scores_tuple = future_score.result()
                img_num_score, llm_text_score, llm_image_score, img_quality_score, ti_rel_score, tq_rel_score, r_tool = scores_tuple

                llm_score = (
                    self.llm_text_factor * llm_text_score
                    + self.llm_image_factor * llm_image_score
                ) / (self.llm_text_factor + self.llm_image_factor)
                llm_score = (llm_score - 1) / 4
                mllm_score = (
                    self.mllm_quality_factor * img_quality_score
                    + self.mllm_ti_factor * ti_rel_score
                    + self.mllm_tq_factor * tq_rel_score
                ) / (
                    self.mllm_quality_factor + self.mllm_ti_factor + self.mllm_tq_factor
                )
                mllm_score = (mllm_score - 1) / 4

                final_score = (
                    self.img_num_factor * img_num_score
                    + self.mllm_factor * mllm_score * img_num_score
                    + self.llm_factor * llm_score
                )

                reward_tensor[i][valid_response_length - 1] = final_score

                reward_extra_info["img_num_score"][i] = img_num_score
                reward_extra_info["llm_text_score"][i] = llm_text_score
                reward_extra_info["llm_image_score"][i] = llm_image_score
                reward_extra_info["img_quality_score"][i] = img_quality_score
                reward_extra_info["ti_rel_score"][i] = ti_rel_score
                reward_extra_info["tq_rel_score"][i] = tq_rel_score
                reward_extra_info["llm_score"][i] = llm_score
                reward_extra_info["mllm_score"][i] = mllm_score
                reward_extra_info["r_tool"][i] = r_tool
                reward_extra_info["score"][i] = final_score

                if "<image>" not in info["vlm_response_str"]:
                    reward_extra_info["ti_rel_score"][i] = 0
                    reward_extra_info["tq_rel_score"][i] = 0
                    reward_extra_info["img_quality_score"][i] = 0
                    reward_extra_info["mllm_score"][i] = 0

                reward_extra_info["stats/diffusion_total"][i] = stats["diffusion"]["total"]
                reward_extra_info["stats/diffusion_success"][i] = stats["diffusion"]["success"]
                reward_extra_info["stats/diffusion_success_rate"][i] = stats["diffusion"]["success_rate"]
                reward_extra_info["stats/search_total"][i] = stats["search"]["total"]
                reward_extra_info["stats/search_success"][i] = stats["search"]["success"]
                reward_extra_info["stats/search_success_rate"][i] = stats["search"]["success_rate"]
                reward_extra_info["stats/code_total"][i] = stats["code"]["total"]
                reward_extra_info["stats/code_success"][i] = stats["code"]["success"]
                reward_extra_info["stats/code_success_rate"][i] = stats["code"]["success_rate"]
                reward_extra_info["stats/edit_total"][i] = stats["edit"]["total"]
                reward_extra_info["stats/edit_success"][i] = stats["edit"]["success"]
                reward_extra_info["stats/edit_success_rate"][i] = stats["edit"]["success_rate"]

                if data_source not in already_print_data_sources:
                    already_print_data_sources[data_source] = 0
                if already_print_data_sources[data_source] < self.num_examine:
                    already_print_data_sources[data_source] += 1
                    print('-' * 20)
                    print(f"[prompt]: \n{info['prompt_str']}")
                    print(f"[response]: \n{info['response_str']}")
                    print(f"[vlm_response_str]: \n{info['vlm_response_str']}")
                    print(f"[stats]: \n{stats}")
                    print(f"[img_num_score]: \n{img_num_score}")
                    print(f"[llm_text_score]: \n{llm_text_score}")
                    print(f"[llm_image_score]: \n{llm_image_score}")
                    print(f"[img_quality_score]: \n{img_quality_score}")
                    print(f"[ti_rel_score]: \n{ti_rel_score}")
                    print(f"[tq_rel_score]: \n{tq_rel_score}")
                    print(f"[llm_score]: \n{llm_score}")
                    print(f"[mllm_score]: \n{mllm_score}")
                    print(f"[score]: \n{final_score}")
                    print('-' * 20)

            except Exception as exc:
                self._handle_processing_error(i, exc, reward_tensor, reward_extra_info, "Score", valid_response_length)

    def process_data_in_pipeline(self, data: DataProto, return_dict=False):
        if 'rm_scores' in data.batch.keys():
            return {"reward_tensor": data.batch['rm_scores']} if return_dict else data.batch['rm_scores']

        num_samples = len(data)
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        already_print_data_sources = {}
        reward_extra_info = self._init_reward_extra_info(num_samples)

        with ThreadPoolExecutor(max_workers=self.vlm_max_workers) as vlm_executor, \
            ThreadPoolExecutor(max_workers=self.score_max_workers) as score_executor:

            future_vlm_to_info = self._submit_vlm_tasks(data, vlm_executor)
            future_score_to_info = self._process_vlm_results(future_vlm_to_info, score_executor, reward_tensor, reward_extra_info)
            self._process_score_results(future_score_to_info, reward_tensor, reward_extra_info, already_print_data_sources)

        for i in range(num_samples):
            reward_extra_info["stats/diffusion_total"][i] *= num_samples
            reward_extra_info["stats/diffusion_success"][i] *= num_samples
            reward_extra_info["stats/search_total"][i] *= num_samples
            reward_extra_info["stats/search_success"][i] *= num_samples
            reward_extra_info["stats/code_total"][i] *= num_samples
            reward_extra_info["stats/code_success"][i] *= num_samples
            reward_extra_info["stats/edit_total"][i] *= num_samples
            reward_extra_info["stats/edit_success"][i] *= num_samples

        assert all(s is not None for s in reward_extra_info["score"])

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info
            }
        else:
            return reward_tensor

    def _handle_processing_error(self, index: int, exc: Exception, reward_tensor: torch.Tensor,
                                reward_extra_info: Dict, stage: str, valid_response_length: int):
        print(f"[ERROR] Sample {index} failed at {stage} stage: {exc}")
        reward_tensor[index][valid_response_length - 1] = 0.0
        for key in reward_extra_info:
            reward_extra_info[key][index] = 0.0

    def __call__(self, data: DataProto, return_dict=False):
        """We will expand this function gradually based on the available datasets"""

        return self.process_data_in_pipeline(data, return_dict)
