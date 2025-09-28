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

from datasets import load_dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor
from imageutils.interleaved_gen import InterleavedGenerator
from PIL import Image
from openai import AzureOpenAI
from io import BytesIO
import base64
from typing import List, Dict, Any
import json
import os
import concurrent
import argparse
import markdown
from qwen_vl_utils import process_vision_info
import re


class BenchJudgeCLient:
    def __init__(
        self,
        base_url: str="",
        api_key: str="",
        api_version: str="",
        model_name: str="gpt-4o",
    ):
        if not base_url or not api_key:
            return
        self.client = AzureOpenAI(
            api_key = api_key,
            api_version=api_version,
            azure_endpoint=base_url
        )
        self.model_name = model_name
        self.system_prompt = 'You are an expert AI performance evaluator. Your task is to objectively assess an AI-generated report based on a specific criterion.'

    @staticmethod
    def _process_image(image: Image.Image):
        if isinstance(image, Image.Image):
            return image.convert("RGB")
    
    @staticmethod
    def _encode_image(image: Image.Image) -> str:
        image = BenchJudgeCLient._process_image(image)
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{img_str}"
    
    def get_test_prompt(self, answer: str, question: str):
        test_prompt = f"INPUT: {question}\nOUTPUT: {answer}"
        return test_prompt

    def _build_messages(
        self, processed_text: str, all_images: List[Image.Image]
    ) -> List[Dict[str, Any]]:
        # This method is now more general, handling a text prompt and a combined list of all images
        content_parts = processed_text.split("<image>")
        message_content = []
        img_counter = 0
        for part in content_parts:
            if part:
                message_content.append({"type": "text", "text": part})
            if img_counter < len(all_images):
                base64_image = BenchJudgeCLient._encode_image(all_images[img_counter])
                message_content.append({"type": "image_url", "image_url": {"url": base64_image}})
                img_counter += 1
        return [{"role": "system", "content": self.system_prompt}, {"role": "user", "content": message_content}]


    def _build_evaluation_prompt(self, question: str, answer_text: str, criterion: str) -> str:
        prompt = f"""
        # CONTEXT
        The full context, including the original prompt's text and images, and the AI's generated report (text and images), is provided below.

        ## Original Benchmark Prompt:
        {question}

        ## AI-Generated Report Text (Images are provided visually):
        {answer_text}

        # TASK
        You must evaluate the "AI-Generated Report" against the single criterion below.
        Your response MUST be a single JSON object with two keys: "score" and "justification". Do not add any text before or after the JSON object.

        ## Scoring Scale:
        - 0: Not Met
        - 1: Partially Met
        - 2: Fully Met or Exceeded

        ## CRITERION TO EVALUATE:
        {criterion}

        ## REQUIRED OUTPUT FORMAT (JSON ONLY):
        {{
        "score": <0, 1, or 2>,
        "justification": "A concise explanation for the score."
        }}
        """
        return prompt
    
    def _judge_answer(
        self, processed_text: str, all_images: List[Image.Image]
    ) -> str:
        if not self.client: raise ConnectionError("Client is not initialized.")
        messages = self._build_messages(processed_text, all_images)
        response = self.client.chat.completions.create(
            model=self.model_name, messages=messages, temperature=0,
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content.strip()
    
    def verify(self, answer: str, question: str, criterion: str, pil_images_list: List[Image.Image], input_images: List[Image.Image]):
        if not self.client:
            print("Error: Evaluator client is not initialized.")
            return -1, "Error: Evaluator client is not initialized."

        # 1. Build the comprehensive prompt for the evaluator LLM
        evaluation_prompt = self._build_evaluation_prompt(question, answer, criterion)

        all_images_for_evaluator = input_images + pil_images_list
        
        # 2. Call the LLM. The images from the answer are passed along with the evaluation prompt.
        try:
            raw_response = self._judge_answer(evaluation_prompt, all_images_for_evaluator)
        except Exception as e:
            print(f"Error during API call: {e}")
            return -1, "Error: Evaluator client is not initialized."

        # 3. Parse the response and return the score
        try:
            # Using response_format={"type": "json_object"} makes this more reliable
            result = json.loads(raw_response)
            score = int(result.get("score"))
            justification = result.get("justification", "No justification provided.")

            if score not in [0, 1, 2]:
                print(f"Warning: Invalid score '{score}' received. Justification: {justification}")
                return -1, "Error: Evaluator client is not initialized." # Indicate invalid score

            # Optional: Print for real-time feedback during execution
            print(f"Criterion: '{criterion[:60]}...' -> Score: {score} | Justification: {justification}")
            
            return score, justification

        except (json.JSONDecodeError, TypeError, ValueError) as e:
            print(f"Error parsing LLM response: {e}. Raw response: {raw_response}")
            return -1, "Error parsing LLM response." # Indicate parsing or type-casting failure 


def generate_html_report(
    sample_index: int,
    prompt: str,
    input_images: List[Image.Image],
    generated_text: str,
    generated_images: List[Image.Image],
    criteria: List[str],
    scores: Dict[str, int],
    justifications: Dict[str, str],
    output_path: str
):
    """
    Generates a single HTML file for one benchmark sample result, now rendering Markdown.
    """
    def pil_to_base64_str(img):
        buffered = BytesIO()
        img.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    # --- HTML and CSS Template (CSS has been updated) ---
    html_template = """
    <!DOCTYPE html><html lang="en"><head>
        <meta charset="UTF-8"><title>Benchmark Result: Sample {sample_index}</title>
        <style>
            body {{ font-family: sans-serif; margin: 0; background: #f8f9fa; color: #212529; }}
            .container {{ max-width: 900px; margin: 20px auto; padding: 20px; background: #fff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #343a40; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; }}
            .section {{ margin-bottom: 30px; }}
            .prompt-box {{ background: #e9ecef; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: monospace; }}
            .image-gallery {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 15px; }}
            .image-gallery img {{ max-height: 200px; width: auto; border: 1px solid #ced4da; border-radius: 5px; }}
            .generated-output img {{ max-width: 65%; height: auto; border-radius: 5px; margin: 15px 0; border: 1px solid #ced4da; }}
            .generated-output table {{ width: 65%; border-collapse: collapse; margin: 20px 0; }}
            .generated-output th, .generated-output td {{ border: 1px solid #dee2e6; padding: 10px; text-align: left; }}
            /* Other styles remain the same */
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; }}
            th, td {{ border: 1px solid #dee2e6; padding: 12px; text-align: left; }}
            th {{ background-color: #f1f3f5; }}
            .score-0 {{ background-color: #ffe3e3; }} .score-1 {{ background-color: #fff9e3; }} .score-2 {{ background-color: #e3f9e3; }}
            .score-cell {{ font-weight: bold; text-align: center; font-size: 1.2em; }}
        </style>
    </head><body><div class="container">
        <h1>Benchmark Result: Sample #{sample_index}</h1>
        <div class="section">
            <h2>Input</h2>
            <div class="prompt-box">{prompt}</div>
            {input_images_html}
        </div>
        <div class="section">
            <h2>Generated Output</h2>
            <div class="generated-output">{generated_content}</div>
        </div>
        <div class="section">
            <h2>Evaluation Rubric</h2>
            <table>
                <thead><tr><th>Criterion</th><th>Score</th><th>Justification</th></tr></thead>
                <tbody>{table_rows}</tbody>
            </table>
        </div>
    </div></body></html>
    """


    input_images_html = ""
    if input_images:
        input_images_html += "<h3>Input Images:</h3><div class='image-gallery'>"
        for img in input_images:
            img_base64 = pil_to_base64_str(img)
            input_images_html += f'<img src="data:image/jpeg;base64,{img_base64}" alt="Input Image">'
        input_images_html += "</div>"

    # --- Populate Generated Content (Now with Markdown rendering) ---
    generated_content = ""
    text_parts = generated_text.split("<image>")
    for i, text_part in enumerate(text_parts):
        if text_part.strip():
            # --- MODIFIED: Convert Markdown to HTML here ---
            # The 'tables' extension is crucial for rendering Markdown tables.
            html_part = markdown.markdown(text_part.strip(), extensions=["fenced_code", "tables", "sane_lists", "nl2br"])
            generated_content += html_part
        
        if i < len(generated_images):
            img_base64 = pil_to_base64_str(generated_images[i])
            generated_content += f'<img src="data:image/jpeg;base64,{img_base64}" alt="Generated Image {i+1}">'
            
    # --- Populate Table Rows ---
    table_rows = ""
    total_score = 0
    max_score = 0
    for i, criterion in enumerate(criteria):
        score = scores.get(f'r{i+1}', -1)
        justification = justifications.get(f'r{i+1}', 'N/A')
        score_class = f"score-{score}" if score != -1 else ""
        table_rows += f'<tr class="{score_class}"><td>{criterion}</td><td class="score-cell">{score}</td><td>{justification}</td></tr>'
        if score != -1:
            total_score += score
            max_score += 2
    
    final_score_text = f"<tr><td colspan='3' style='text-align:right; font-weight:bold;'>Final Score: {total_score} / {max_score}</td></tr>"
    table_rows += final_score_text
    
    # --- Finalize and Write HTML ---
    final_html = html_template.format(
        sample_index=sample_index, prompt=prompt, input_images_html=input_images_html,
        generated_content=generated_content, table_rows=table_rows
    )
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(final_html)
    print(f"  - HTML report saved to: {output_path}")



def build_messages(
        processed_text: str, all_images: List[Image.Image], system_prompt: str
    ) -> List[Dict[str, Any]]:
        # This method is now more general, handling a text prompt and a combined list of all images
        content_parts = processed_text.split("<image>")
        message_content = []
        img_counter = 0
        for part in content_parts:
            if part:
                message_content.append({"type": "text", "text": part})
            if img_counter < len(all_images):
                # base64_image = BenchJudgeCLient._encode_image(all_images[img_counter])
                message_content.append({"type": "image"})
                img_counter += 1
        return [{"role": "system", "content": [{"type": "text", "text": system_prompt}]}, {"role": "user", "content": message_content}]


def inference(model, tokenizer, processor, system_prompt: str, query: str, input_images: List[Image.Image]):
    sampling_params = SamplingParams(max_tokens=1024*16)
    inter_gen = InterleavedGenerator()

    messages = build_messages(query, input_images, system_prompt)

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = {"prompt": text, "multi_modal_data": {"image": input_images}}


    outputs = model.generate(inputs, sampling_params=sampling_params)

    output_text = outputs[0].outputs[0].text
    processed_text, image_list, _ = inter_gen.interleaved_generation(output_text)

    return output_text, processed_text, image_list
            

def main(model: str, input_file: str, output_dir: str, tensor_parallel_size: int = 4):
    """
    Main function to run the entire evaluation pipeline.
    """
    json_output_dir = os.path.join(output_dir, "json_results")
    html_output_dir = os.path.join(output_dir, "html_reports")
    os.makedirs(json_output_dir, exist_ok=True)
    os.makedirs(html_output_dir, exist_ok=True)
    json_output_file = os.path.join(json_output_dir, "evaluation_results.json")

    with open("system_prompts/mllm_sp/sp1.txt", 'r', encoding='utf-8') as f:
        system_prompt = f.read()

    print(f"Loading Model: {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    processor = AutoProcessor.from_pretrained(model, min_pixels=200*200, max_pixels=1024*1024)
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
    )
    evaluator = BenchJudgeCLient()


    with open(input_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)


    all_results = []

    for i, item in enumerate(tqdm(dataset, desc="Evaluating Benchmarks")):
        sample_index = i + 1
        prompt = item.get("prompt")
        if not prompt:
            print(f"Skipping item {i} due to missing 'prompt' key.")
            continue

        input_image_paths = item.get("images", [])
        input_images = []
        for path in input_image_paths:
            try:
                input_images.append(Image.open(path).convert("RGB"))
            except FileNotFoundError:
                print(f"Warning: Input image not found at {path} for sample {sample_index}. Skipping this image.")
        
            
        criteria = [item[f'r{j}'] for j in range(1, 11) if f'r{j}' in item]

        output_text, generated_text, generated_images = inference(llm, tokenizer, processor, system_prompt, query=prompt, input_images=input_images)
        
        scores = {}
        justifications = {}
        print(f"--- Judging item {i+1}/{len(dataset)} ---")
        for j, criterion in enumerate(criteria, 1):
            score, justification = evaluator.verify(
                answer=generated_text,
                question=prompt,
                criterion=criterion,
                pil_images_list=generated_images,
                input_images=input_images
            )
            scores[f'r{j}'] = score
            justifications[f'r{j}'] = justification
        
        result_item = {
            "sample_index": sample_index,
            "prompt": prompt,
            "answer": output_text,
            "scores": scores,
            "justifications": justifications
        }
        all_results.append(result_item)

        html_report_path = os.path.join(html_output_dir, f"report_sample_{sample_index:03d}.html")
        generate_html_report(
            sample_index=sample_index, prompt=prompt, input_images=input_images, generated_text=generated_text,
            generated_images=generated_images, criteria=criteria, scores=scores,
            justifications=justifications, output_path=html_report_path,
        )
        
        if (i + 1) % 5 == 0 or (i + 1) == len(dataset):
             with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=4, ensure_ascii=False)
             print(f"\n--- Progress saved to {json_output_file} ({i+1}/{len(dataset)} items processed) ---\n")

    print(f"Evaluation complete. All results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the evaluation pipeline.")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B", help="Path to the model.")
    parser.add_argument("--input_file", type=str, default="benchmark/mm_input.json", help="Path to the input JSON file.")
    parser.add_argument("--output_dir", type=str, default="eval_results/Bench/QwenVL-7B-mm", help="Path to the output JSON file.")
    parser.add_argument("--tensor_parallel_size", type=int, default=4, help="Tensor parallel size.")
    args = parser.parse_args()

    main(args.model, args.input_file, args.output_dir, args.tensor_parallel_size)