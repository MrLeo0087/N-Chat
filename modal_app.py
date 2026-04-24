import os
import re
import time
import modal
import concurrent.futures
import threading
import traceback
import base64
import numpy as np
import cv2
from typing import Dict, Any, List

# ============================================================================
# CONFIGURATION
# ============================================================================

os.environ["PADDLEX_PADDLE_INFERENCE_PARALLEL"] = "False"

APP_NAME = "paddleocr-vl-modal"
GPU_TYPE = "A10G"

ENV_VARS = {
    "PADDLE_INF_SERIAL": "1",
    "PADDLE_INF_NUM_THREADS": "1",
    "PADDLE_INF_THREADS": "1",
    "PADDLE_PDX_VLM_PARALLEL": "0",
    "OMP_NUM_THREADS": "1",
    "FLAGS_enable_pir_api": "0",
    "FLAGS_enable_pir_in_executor": "0",
    "FLAGS_ir_optim_cache_disable": "1",
    # Persistent model storage
    "PADDLE_HOME": "/models/paddle",
    "PADDLEOCR_BASE_DIR": "/models/paddleocr",
    "XDG_CACHE_HOME": "/models/cache",
}

# ============================================================================
# MODAL APP SETUP
# ============================================================================

app = modal.App(APP_NAME)

# Create optimized container image
image = (
    modal.Image.from_registry(
        "paddlepaddle/paddle:3.0.0-gpu-cuda11.8-cudnn8.9-trt8.6",
    )
    .apt_install(
        "tini", "curl", "libgl1", "libglib2.0-0",
        "build-essential", "clang", "cmake", "git",
        "libgomp1", "libsm6", "libxext6", "libxrender-dev",
        "wget", "libgl1-mesa-glx", "libsm6", "libxrender1", "libxext6"
    )
    # Set environment variables for build-time CPU mode
    .env({
        "CUDA_VISIBLE_DEVICES": "",
        "FLAGS_cpu_deterministic": "true",
        "DISABLE_MODEL_SOURCE_CHECK": "True",
        "OMP_NUM_THREADS": "1",
    })
    # Upgrade pip and clean up problematic packages
    .run_commands(
        "pip uninstall -y paddlepaddle paddlepaddle-gpu paddlex "
        "paddleocr || true",
        "pip uninstall -y PyYAML urllib3 typing-extensions "
        "opencv-python opencv-python-headless || true",
    )
    # Install compatible versions in the correct order
    .run_commands(
        "pip install --no-cache-dir numpy==1.26.4",
        "pip install --no-cache-dir shapely==2.0.6 pyclipper==1.3.0.post5",
        "pip install --no-cache-dir opencv-python-headless==4.10.0.84",
        "pip install --no-cache-dir scipy==1.12.0",
        "pip install --no-cache-dir pillow==10.4.0",
        "pip install --no-cache-dir rapidfuzz==3.9.3 "
        "python-Levenshtein==0.25.1",
        "pip install --no-cache-dir lanms-neo==1.0.2 visualdl==2.5.3",

        # Install PaddlePaddle GPU
        "pip install --no-cache-dir 'paddlepaddle-gpu>=3.0.0b2' "
        "-i https://www.paddlepaddle.org.cn/packages/stable/cu118/",

        "pip install --no-cache-dir --ignore-installed "
        "'paddleocr[doc-parser]>=2.9.1'",
        "pip install --no-cache-dir --ignore-installed 'paddlex>=3.3.6'",
    )
    # Install FastAPI and monitoring stack
    .run_commands(
        "pip install --no-cache-dir fastapi==0.128.0 "
        "uvicorn[standard]==0.30.1",
        'pip install --no-cache-dir "python-multipart>=0.0.5"',
        "pip install --no-cache-dir pydantic==2.12.5 pydantic-settings==2.5.2",
        "pip install --no-cache-dir prometheus-client==0.22.0 "
        "prometheus-fastapi-instrumentator==7.1.0",
        "pip install --no-cache-dir structlog==24.4.0 orjson==3.10.6",
    )
    # Create runtime verification script
    .run_commands(
        "mkdir -p /app/scripts",
        "echo '#!/bin/bash\\n"
        "export CUDA_VISIBLE_DEVICES=0\\n"
        "python -c \"import paddle; print(\\\"PaddlePaddle: \\\" + "
        "paddle.__version__)\"\\n"
        "python -c \"import paddleocr; print(\\\"PaddleOCR: \\\" + "
        "paddleocr.__version__)\"' > /app/scripts/verify_deps.sh",
        "chmod +x /app/scripts/verify_deps.sh",
    )
    # Reset environment variables for runtime
    .env({
        **ENV_VARS,
        "CUDA_VISIBLE_DEVICES": "0",
        "OMP_NUM_THREADS": "1",
    })
    .add_local_python_source("modal_app", "api")
)

# Volume for model cache
volume = modal.Volume.from_name("paddleocr-models", create_if_missing=True)

# ============================================================================
# OCR SERVICE CLASS
# ============================================================================


@app.cls(
    image=image,
    gpu=GPU_TYPE,
    cpu=4.0,
    memory=16384,
    max_containers=5,
    scaledown_window=150,
    enable_memory_snapshot=False,
    volumes={"/models": volume},
    timeout=300,
)
class PaddleOCRService:
    """
    PaddleOCR service with scale-to-zero after 2 minutes

    Features:
    - Scales to zero when idle (no cost)
    - Fast cold start: 3-5 seconds with memory snapshots
    - Warm inference: 200-500ms
    - Automatic scaling up to handle load
    """

    @modal.enter()
    def load_model(self):
        """
        Runs once when container starts (not per request!)
        This is for avoiding per-request cold starts
        """
        import os
        import threading
        # Force disable PIR before importing paddle
        os.environ["FLAGS_enable_pir_api"] = "0"
        os.environ["FLAGS_enable_pir_in_executor"] = "0"
        os.environ["FLAGS_ir_optim_cache_disable"] = "1"

        start_time = time.time()

        import paddle
        import numpy as np
        from paddleocr import PaddleOCRVL

        # Monkey-patch masked_scatter for Paddle 3.0 compatibility
        try:
            _orig_masked_scatter = paddle.Tensor.masked_scatter
            def patched_masked_scatter(self, mask, value):
                if mask.dtype != paddle.bool:
                    mask = mask.cast(paddle.bool)
                return _orig_masked_scatter(self, mask, value)
            paddle.Tensor.masked_scatter = patched_masked_scatter
            print("INFO: Patching paddle.Tensor.masked_scatter for Paddle 3.0")
        except Exception as e:
            print(f"WARN: Failed to patch masked_scatter: {e}")

        print("=" * 60)
        print("Container starting - loading PaddleOCR-VL model...")
        print("=" * 60)

        # Disable PIR API for stability (redundant but safe)
        try:
            paddle.set_flags({
                "FLAGS_enable_pir_api": 0,
                "FLAGS_enable_pir_in_executor": 0,
                "FLAGS_ir_optim_cache_disable": 1
            })
        except Exception:
            pass

        try:
            print(f"Paddle device: {paddle.device.get_device()}")
        except Exception as e:
            print(f"Warning checking device: {e}")

        self.ocr = PaddleOCRVL(
            # doc_orientation_classify_model_name="PP-LCNet_x1_0_doc_ori",
            doc_unwarping_model_name="UVDoc",
            layout_detection_model_name="PP-DocLayoutV2",
            use_layout_detection=True,
            use_doc_orientation_classify=False,  # Set False because makes bad prediction in Nepali
            use_doc_unwarping=True
        )

        print("Warming up model with dummy inference...")
        try:
            dummy_path = "/tmp/warmup.jpg"
            import cv2
            dummy_img = 255 * np.ones((640, 640, 3), dtype=np.uint8)
            cv2.imwrite(dummy_path, dummy_img)

            _ = self.ocr.predict(
                dummy_path, use_queues=False, prompt_label="ocr"
            )
            print("Warmup successful!")

            # Cleanup warmup file
            if os.path.exists(dummy_path):
                os.unlink(dummy_path)
        except Exception as e:
            print(f"Warmup warning: {e}")
            import traceback
            traceback.print_exc()

        elapsed = time.time() - start_time

        print("=" * 60)
        print(f"Model loaded and warmed up in {elapsed:.2f}s")
        print(f"GPU device: {paddle.device.get_device()}")
        print("=" * 60)

    def _decode_image(self, image_data: Any):
        """
        Robustly decode image data (base64 str or bytes) into OpenCV format.
        """
        import base64
        import numpy as np
        import cv2
        import re

        try:
            if isinstance(image_data, bytes):
                nparr = np.frombuffer(image_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if img is None:
                    raise ValueError("OpenCV could not decode image bytes")
                return img

            if not isinstance(image_data, str):
                raise ValueError(f"Unsupported image type: {type(image_data)}")

            clean_base64 = image_data.strip()

            if clean_base64.startswith("data:image"):
                match = re.search(r",(.+)$", clean_base64)
                if match:
                    clean_base64 = match.group(1)
                else:
                    comma_idx = clean_base64.find(",")
                    if comma_idx != -1:
                        clean_base64 = clean_base64[comma_idx+1:]

            try:
                image_bytes = base64.b64decode(clean_base64)
            except Exception as e:
                raise ValueError(f"Failed to decode base64 string: {str(e)}")

            nparr = np.frombuffer(image_bytes, np.uint8)
            if nparr.size == 0:
                raise ValueError("Image buffer is empty")

            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError(
                    "OpenCV could not decode image (invalid or corrupt data)"
                )

            return img

        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Decoding error: {str(e)}")

    def _make_serializable(self, obj):
        """Recursively convert objects to JSON-serializable types."""
        import numpy as np
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return self._make_serializable(obj.tolist())
        elif isinstance(obj, (list, tuple)):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, set):
            return [self._make_serializable(item) for item in list(obj)]
        elif isinstance(obj, dict):
            return {str(k): self._make_serializable(v) for k, v in obj.items()}
        elif hasattr(obj, "tolist"):
            return self._make_serializable(obj.tolist())
        elif hasattr(obj, "__dict__"):
            return self._make_serializable(vars(obj))

        if obj is None or isinstance(obj, (int, float, str, bool)):
            return obj

        try:
            return str(obj)
        except Exception:
            return f"Unserializable: {type(obj)}"

    def _clean_text(self, text: str, strip_html: bool = True) -> str:
        """Clean up internal PaddleOCR tags and optionally strip HTML."""
        if not text:
            return ""

        text = text.replace("<fcel>", " | ")
        text = text.replace("<nl>", "\n")
        text = text.replace("<frow>", "\n")

        if strip_html:
            text = re.sub(r'<[^>]+>', '', text)

        return text.strip()

    def _generate_markdown(self, raw_result: Dict[str, Any]) -> str:
        """Extract or generate markdown content from OCR results."""
        markdown_content = raw_result.get('markdown', "")
        if markdown_content:
            # Don't strip HTML from binary/extracted markdown as it likely contains structure
            return self._clean_text(markdown_content, strip_html=False)

        # Fallback: build markdown from parsing results
        p_list = []
        if isinstance(raw_result, dict):
            p_list = raw_result.get('parsing_res_list')
            if p_list is None:
                p_list = []

            if not p_list:
                l_res = raw_result.get('layoutParsingResults')
                if l_res and len(l_res) > 0:
                    res = l_res[0].get('prunedResult', {})
                    p_list = res.get('parsing_res_list', [])

        if not p_list:
            return ""

        md_lines = []
        for item in p_list:
            item_data = (item if isinstance(item, dict)
                         else (vars(item) if hasattr(item, '__dict__') else {}))
            content = (item_data.get('content') or 
                       item_data.get('block_content', ""))
            if content:
                content = self._clean_text(content)
                block_type = item_data.get('type', 'text').lower()
                if block_type == 'header':
                    md_lines.append(f"# {content}")
                elif block_type == 'title':
                    md_lines.append(f"## {content}")
                elif block_type == 'table':
                    # Keep HTML structure for tables
                    md_lines.append(self._clean_text(content, strip_html=False))
                else:
                    md_lines.append(self._clean_text(content, strip_html=True))
                md_lines.append("") # Spacer
        return "\n".join(md_lines)

    def _extract_texts(self, final_result: Dict[str, Any]) -> list:
        """
        Extract structured text objects from the final result dictionary.
        """
        texts_list = []
        try:
            if 'ocr_result' in final_result:
                ocr_res = final_result.get('ocr_result', {})
                lines = ocr_res.get('lines')
                if lines is not None and len(lines) > 0:
                    for line in lines:
                        conf = line.get('score')
                        if conf is None:
                            conf = line.get('confidence')
                        if conf is None:
                            conf = 1.0

                        raw_text = line.get('text')
                        if raw_text:
                            # Clean and split in case \n or <nl> are present
                            cleaned_text = self._clean_text(str(raw_text))
                            bbox = line.get('bbox')
                            for sub_line in cleaned_text.split('\n'):
                                if sub_line.strip():
                                    texts_list.append({
                                        "text": sub_line.strip(),
                                        "bbox": bbox,
                                        "confidence": float(conf) if hasattr(conf, "__float__") or isinstance(conf, (int, float)) else 1.0
                                    })
                    return texts_list

            p_list = final_result.get('parsing_res_list')
            if p_list is None or len(p_list) == 0:
                l_res = final_result.get('layoutParsingResults')
                if l_res and len(l_res) > 0:
                    res = l_res[0].get('prunedResult', {})
                    p_list = res.get('parsing_res_list')

            if p_list:
                for item in p_list:
                    item_data = (item if isinstance(item, dict)
                                 else (vars(item) if hasattr(item, '__dict__') else {}))
                    text_content = (item_data.get('content') or
                                    item_data.get('block_content', ''))

                    if text_content:
                        text_content = self._clean_text(str(text_content))
                        bbox = item_data.get('bbox')
                        if bbox is None:
                            bbox = item_data.get('block_bbox')

                        conf = item_data.get('confidence')
                        if conf is None:
                            conf = item_data.get('score')
                        if conf is None:
                            conf = 1.0
                        for line in text_content.split('\n'):
                            if line.strip():
                                texts_list.append({
                                    "text": line.strip(),
                                    "bbox": bbox,
                                    "confidence": conf
                                })
        except Exception as e:
            print(f"DEBUG: Parse warning: {e}")
        return texts_list

    def _encode_image(self, img_array) -> str:
        """Convert OpenCV image array to base64 string."""
        import cv2
        import base64
        if img_array is None:
            return None
        try:
            _, buffer = cv2.imencode('.jpg', img_array, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            return base64.b64encode(buffer).decode('utf-8')
        except Exception as e:
            print(f"Error encoding image: {e}")
            return None

    @modal.method()
    def predict(
        self,
        image_data: Any,
        prompt: str = "ocr",
        use_layout_detection: bool = True,
        use_doc_orientation_classify: bool = False,
        use_doc_unwarping: bool = True,
        layout_merge_bboxes_mode: str = "small",
        return_markdown: bool = True,
    ) -> Dict[str, Any]:
        """Run OCR prediction on a single image (Modal Entrypoint)."""
        import time
        try:
            # Decode image (bytes, base64, or filepath)
            img = self._decode_image(image_data)

            inf_start = time.time()
            result = self.ocr.predict(
                img,
                use_queues=False,
                prompt_label=prompt,
                use_layout_detection=use_layout_detection,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                use_doc_orientation_classify=use_doc_orientation_classify,
                use_doc_unwarping=use_doc_unwarping
            )
            inf_time_ms = (time.time() - inf_start) * 1000

            # Extract the first element if the model returns a list
            raw_result = result[0] if isinstance(result, list) and result else result

            # Encode final image to base64
            processed_img_b64 = None
            if isinstance(raw_result, dict):
                prep_res = raw_result.get('doc_preprocessor_res', {})
                if not isinstance(prep_res, dict):
                    prep_res = {}

                # Safely select the best available image for output
                final_img = prep_res.get('output_img')
                if final_img is None:
                    final_img = raw_result.get('output_img')
                if final_img is None:
                    final_img = prep_res.get('rot_img')
                if final_img is None:
                    final_img = raw_result.get('rot_img')
                if final_img is None:
                    final_img = img

                processed_img_b64 = self._encode_image(final_img)

            # Generate markdown (if requested)
            markdown_content = ""
            if return_markdown:
                markdown_content = self._generate_markdown(raw_result)

            # Clean large arrays from raw_result to make JSON serialization safe
            if isinstance(raw_result, dict):
                raw_result.pop('doc_preprocessor_res', None)
                for key in list(raw_result.keys()):
                    k_lower = key.lower()
                    if any(x in k_lower for x in ['image', 'map', 'mask', 'feature']):
                        val = raw_result.get(key)
                        raw_result[key] = f"Removed ({getattr(val, 'shape', 'unknown')})"

            # Make everything JSON-serializable
            final_result = self._make_serializable(raw_result)

            # Extract text lines & confidences
            texts = self._extract_texts(final_result)

            return {
                "success": True,
                "texts": texts,
                "count": len(texts),
                "inference_time_ms": round(inf_time_ms, 2),
                "markdown": markdown_content if return_markdown else None,
                "processed_image": processed_img_b64
            }

        except Exception as e:
            err_msg = f"{type(e).__name__}: {str(e)}"
            print(f"Prediction error: {err_msg}")
            traceback.print_exc()
            return {
                "success": False,
                "error": err_msg,
                "count": 0
            }
