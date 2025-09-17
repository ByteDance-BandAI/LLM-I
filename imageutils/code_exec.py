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

import subprocess
import sys
from PIL import Image
import io
import re
import codecs


def sanitize_code(code: str) -> str:
    code = re.sub(r"^```(?:python)?", "", code.strip(), flags=re.IGNORECASE)
    code = re.sub(r"```$", "", code.strip())

    try:
        code = codecs.decode(code, 'unicode_escape')
    except Exception as e:
        print(f"[WARNING] Code sanitize failed: {e}")
        code = code.replace("\\n", "\n").replace("\\t", "\t").replace("\\\"", "\"").replace("\\'", "'")

    return code


def codeexec(llmcode):
    footer_code = """\n\n
try:
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    plt.close('all')
    image_bytes = buffer.getvalue()
    sys.stdout.buffer.write(image_bytes)
    sys.stdout.flush()

except Exception as e:
    sys.stderr.write(f"fail: {e}")
    sys.stderr.flush()
    sys.exit(1)
    """

    head_code = """import sys\nimport io\nimport matplotlib.pyplot as plt\nimport pandas as pd\nimport seaborn as sns\nimport numpy as np\nplt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei', 'Arial Unicode MS']\nplt.rcParams['axes.unicode_minus'] = False\n\n"""

    try:
        llmcode = sanitize_code(llmcode)
        all_code = head_code + llmcode + footer_code
        completed_process = subprocess.run(
            [sys.executable, "-c", all_code],
            capture_output=True,
            text=False,
            check=True,
            timeout=10,
        )

        image_bytes_from_subprocess = completed_process.stdout

        if not image_bytes_from_subprocess:
            raise ValueError("No images returned")

        image = Image.open(io.BytesIO(image_bytes_from_subprocess))
        return image, ""

    except subprocess.TimeoutExpired:
        print("[WARNING] Code execution timeout.")
        return None, "Code execution timeout."
    except subprocess.CalledProcessError as e:
        print("[WARNING] Code execution failed.")
        print("Error message:")
        print(e.stderr)
        return None, f"Code execution failed with error message: {e.stderr}"
    except Exception as e:
        print(f"[WARNING] Code execution failed with error message: {e}")
        return None, f"Code execution failed with error message: {e}"
