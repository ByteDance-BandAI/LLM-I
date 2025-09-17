import os
import re
import string
import time

from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import ast
from typing import Dict, Any, Optional, Set

from verl.utils.reward_score.judge_client import LLMJudgeClient, MLLMJudgeClient


LLM_JUDGE_CLIENT = LLMJudgeClient(
    base_url=os.environ.get("LLM_JUDGE_BASE_URL", ""),
    model_name=os.environ.get("LLM_JUDGE_MODEL_NAME", None),
)


MLLM_JUDGE_CLIENT = MLLMJudgeClient(
    base_url=os.environ.get("MLLM_JUDGE_BASE_URL", ""),
    model_name=os.environ.get("MLLM_JUDGE_MODEL_NAME", None),
)


def parse_image_tag(dict_string: str) -> Optional[Dict[str, Any]]:
    if "'source': 'code'" in dict_string:
        code_pattern = r"^\s*{'source': 'code', 'description': '([^']*)', 'params': {'code': '(.*)'}}\s*$"
        match = re.search(code_pattern, dict_string, re.DOTALL)   
        if match:
            description, code = match.groups()
            return {
                "source": "code",
                "description": description,
                "params": {"code": code}
            }
    try:
        parsed_dict = ast.literal_eval(dict_string)
        if isinstance(parsed_dict, dict):
            return parsed_dict
        return None
    except:
        return None


def is_image_valid(content: str, tag_len: int, input_image_num: int = 0) -> bool:
    data = parse_image_tag(content)

    if not data:
        return False
    if not isinstance(data, dict):
        return False
    required_keys = {"source", "description", "params"}
    if set(data.keys()) != required_keys:
        return False
    if not isinstance(data.get("description"), str) or not isinstance(
        data.get("params"), dict
    ):
        return False
    source = data.get("source")
    params = data.get("params")
    if source == "diffusion":
        return isinstance(params.get("prompt"), str) and set(params.keys()) == {
            "prompt"
        }
    elif source == "search":
        return isinstance(params.get("query"), str) and set(params.keys()) == {"query"}
    elif source == "code":
        return isinstance(params.get("code"), str) and set(params.keys()) == {"code"}
    elif source == "edit":
        return (
            isinstance(params.get("img_index"), int)
            and (params.get("img_index") >= 0)
            and params.get("img_index") < input_image_num + tag_len
            and isinstance(params.get("prompt"), str)
            and set(params.keys()) == {"img_index", "prompt"}
        )
    return False


def image_num_score(vlm_response: str, image_num: int) -> float:
    valid_count = vlm_response.count("<image>")

    if image_num == -1:
        score = 1 if valid_count == 0 else 0
    elif image_num == 0:
        score = 1
    elif image_num == 99:
        score = 1 if valid_count > 0 else 0
    else:
        score = 1 if valid_count == image_num else valid_count / image_num
        if score > 1.0:
            score = max(0, 1.0 - (valid_count - image_num) * 0.3)

    return score


def image_num_score_eval(vlm_response: str, image_num: int) -> int:
    valid_count = vlm_response.count("<image>")

    if image_num == -1:
        return 1 if valid_count == 0 else 0
    if image_num == 0:
        return 1
    if image_num == 99:
        if valid_count > 0:
            return 1
        return 0
    return 1 if image_num == valid_count else 0

def extract_success_tool(stats: dict) -> set:
    return {k for k, v in stats.items() if v['success'] > 0}

def extract_attempted_tools(stats: dict) -> set:
    return {k for k, v in stats.items() if v['total'] > 0}

def calculate_r_tool(
    required_tools: Set[str],
    attempted_tools: Set[str],
    succeeded_tools: Set[str],
    tool_difficulty_weights: Dict[str, float] = {"code": 0.4, "edit": 0.3, "search": 0.2, "diffusion": 0.1},
    beta: float = 0.5,
    tools_are_optional: bool = True,
) -> float:
    if not required_tools:
        if tools_are_optional:
            return 0.0
        else:
            return 1.0 if not attempted_tools else 0.0

    denominator_motivation = sum(tool_difficulty_weights.get(t, 0) for t in required_tools)
    if denominator_motivation == 0:
        return 1.0 if not attempted_tools else 0.0

    numerator_attempt = sum(
        tool_difficulty_weights.get(t, 0) for t in required_tools.intersection(attempted_tools)
    )
    r_attempt = numerator_attempt / denominator_motivation

    numerator_success = sum(
        tool_difficulty_weights.get(t, 0) for t in required_tools.intersection(succeeded_tools)
    )
    r_success = numerator_success / denominator_motivation

    r_tool = beta * r_success + (1 - beta) * r_attempt
    return r_tool

def compute_score(
    solution_str: str,
    image_num: int,
    data_source: str,
    vlm_response_str: str,
    pil_img_list: list[Image.Image],
    mm_input_len: int,
    tool_stats: dict,
    extra_info=None,
) -> tuple[float, str]:
    if data_source in ["interleaved", "interleaved_mm"]:
        if extra_info["split"] == "train":
            # rule-based image num score
            img_num_score = image_num_score(vlm_response_str, image_num)
        else:
            img_num_score = image_num_score_eval(vlm_response_str, image_num)
        
        succeeded_tools = extract_success_tool(tool_stats)
        attempted_tools = extract_attempted_tools(tool_stats)
        required_tools = set(extra_info['tools'])

        r_tool = calculate_r_tool(required_tools, attempted_tools, succeeded_tools)
        
        with ThreadPoolExecutor(max_workers=2) as executor:
            llm_future = executor.submit(
                LLM_JUDGE_CLIENT.verify,
                answer=solution_str, 
                question=extra_info["question"]
            )
            if '<image>' not in vlm_response_str:
                img_quality_score, ti_rel_score, tq_rel_score = 5, 5, 5
            elif vlm_response_str.count("<image>") > 10:
                img_quality_score, ti_rel_score, tq_rel_score = 1, 1, 1
            else:
                mllm_future = executor.submit(
                    MLLM_JUDGE_CLIENT.verify,
                    answer=vlm_response_str,
                    question=extra_info["question"],
                    pil_images_list=pil_img_list,
                )

                try:
                    img_quality_score, ti_rel_score, tq_rel_score = mllm_future.result()
                except Exception as e:
                    print("mllm_future error:", e)
                    img_quality_score, ti_rel_score, tq_rel_score = 1, 1, 1
            try:
                llm_text_score, llm_image_score = llm_future.result()
                if vlm_response_str.replace("<image>", "").strip() == "":
                    llm_text_score = 1
            except Exception as e:
                print("llm_future error:", e)
                llm_text_score, llm_image_score = 1, 1

        return (
            img_num_score,
            llm_text_score,
            llm_image_score,
            img_quality_score,
            ti_rel_score,
            tq_rel_score,
            r_tool
        )