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

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from imageutils.interleaved_gen import InterleavedGenerator
from PIL import Image
from io import BytesIO
import base64
from typing import List
import os
import argparse
import markdown



def generate_html_report(
    prompt: str,
    generated_text: str,
    generated_images: List[Image.Image],
    output_path: str,
):
    """
    Generates a single HTML file for one inference result, rendering Markdown.
    """

    def pil_to_base64_str(img):
        buffered = BytesIO()
        img.convert("RGB").save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode()

    # --- HTML and CSS Template ---
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Inference Result</title>
        <style>
            body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; margin: 0; background-color: #f8f9fa; color: #212529; }}
            .container {{ max-width: 900px; margin: 20px auto; padding: 20px; background-color: #ffffff; border-radius: 8px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            h1, h2 {{ color: #343a40; border-bottom: 2px solid #dee2e6; padding-bottom: 10px; }}
            .section {{ margin-bottom: 30px; }}
            .prompt-box {{ background-color: #e9ecef; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-family: "Courier New", Courier, monospace; }}
            .generated-output img {{ max-width: 70%; height: auto; border-radius: 5px; margin: 15px 0; border: 1px solid #ced4da; }}
            /* Styles for rendered Markdown content */
            .generated-output h1, .generated-output h2, .generated-output h3, .generated-output h4 {{ border-bottom: none; padding-bottom: 0; margin-top: 24px; margin-bottom: 10px; }}
            .generated-output p {{ line-height: 1.7; margin: 1em 0; }}
            .generated-output table {{ width: 70%; border-collapse: collapse; margin: 20px 0; }}
            .generated-output th, .generated-output td {{ border: 1px solid #dee2e6; padding: 10px; text-align: left; }}
            .generated-output th {{ background-color: #f1f3f5; }}
            .generated-output ul, .generated-output ol {{ padding-left: 20px; }}
            .generated-output li {{ margin-bottom: 8px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Inference Result</h1>
            <div class="section">
                <h2>Input Prompt</h2>
                <div class="prompt-box">{prompt}</div>
            </div>
            <div class="section">
                <h2>Generated Output</h2>
                <div class="generated-output">{generated_content}</div>
            </div>
        </div>
    </body>
    </html>
    """

    # --- Populate Generated Content (Now with Markdown rendering) ---
    generated_content = ""
    text_parts = generated_text.split("<image>")
    for i, text_part in enumerate(text_parts):
        if text_part.strip():
            # Convert Markdown to HTML
            html_part = markdown.markdown(
                text_part.strip(),
                extensions=["fenced_code", "tables", "sane_lists", "nl2br"],
            )
            generated_content += html_part

        if i < len(generated_images):
            img_base64 = pil_to_base64_str(generated_images[i])
            generated_content += f'<img src="data:image/jpeg;base64,{img_base64}" alt="Generated Image {i+1}">'

    # --- Finalize and Write HTML ---
    final_html = html_template.format(
        prompt=prompt,
        generated_content=generated_content,
    )
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(final_html)
    print(f"HTML report saved to: {output_path}")


def inference(model, tokenizer, system_prompt: str, query: str):
    sampling_params = SamplingParams(max_tokens=1024 * 24)
    inter_gen = InterleavedGenerator()

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]
    formatted_prompts = []
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    formatted_prompts.append(text)

    outputs = model.generate(formatted_prompts, sampling_params)

    output_text = outputs[0].outputs[0].text
    processed_text, image_list, _ = inter_gen.interleaved_generation(output_text)

    return output_text, processed_text, image_list


def main(model: str, prompt: str, output: str, tensor_parallel_size: int = 4):
    """
    Main function to run inference on a single prompt and save as HTML.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output), exist_ok=True)

    sp_path = "system_prompts/training/llmi.txt"

    with open(sp_path, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    print(f"Loading model: {model}...")
    tokenizer = AutoTokenizer.from_pretrained(model)
    llm = LLM(
        model=model,
        tensor_parallel_size=tensor_parallel_size,
    )
    
    print(f"Running inference on prompt: {prompt}")
    output_text, generated_text, generated_images = inference(
        llm, tokenizer, system_prompt, query=prompt
    )

    generate_html_report(
        prompt=prompt,
        generated_text=generated_text,
        generated_images=generated_images,
        output_path=output,
    )

    print(f"Inference complete. Result saved to {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single prompt.")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Instruct-2507",
        help="Path to the model.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Prepare a market research report on automobiles, including data analysis of prominent brands, future trends, and an introduction to the latest products.",
        help="The prompt to run inference on.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="eval_results/result.html",
        help="Path to the output HTML file.",
    )
    parser.add_argument(
        "--tensor_parallel_size", type=int, default=4, help="Tensor parallel size."
    )
    args = parser.parse_args()

    main(args.model, args.prompt, args.output, args.tensor_parallel_size)
