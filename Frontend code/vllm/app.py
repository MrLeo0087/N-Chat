import modal
import os
import time
import httpx
import json
from datetime import datetime

# --- CONFIGURATION ---
WORKSPACE = "info-nchat" 
EXISTING_VOLUME_NAME = "paddle-ocr-weights"
APP_NAME = "paddleocr-vl-vllm-deployment"

# --- IMAGES ---
paddle_image = (
    modal.Image.from_registry("paddlepaddle/paddle:3.3.0-gpu-cuda13.0-cudnn9.13")
    .pip_install("paddleocr[doc-parser]", "fastapi[standard]", "pydantic-settings", "httpx", "numpy")
)

vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("vllm==0.14.0", "openai", "fastapi", "torch>=2.0.0")
    .env({
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    })
)

app = modal.App(APP_NAME)
existing_volume = modal.Volume.from_name(EXISTING_VOLUME_NAME)

# --- vLLM BACKEND ---
@app.cls(
    image=vllm_image,
    gpu="A10G", 
    scaledown_window=300,
    volumes={"/data": existing_volume},
    container_idle_timeout=300,
)
class VllmBackend:
    @modal.asgi_app(label="vllm-ocr-api")
    def server(self):
        from vllm.entrypoints.openai.api_server import router
        from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
        from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from fastapi import FastAPI, Request
        import torch

        model_name = "PaddleOCR-VL-0.9B"
        local_model_path = "/data/model-flat"

        # OPTION 1: Increase max_model_len to match PaddleOCR's request
        engine_args = AsyncEngineArgs(
            model=local_model_path,
            served_model_name=[model_name],
            trust_remote_code=True,
            enforce_eager=False,
            gpu_memory_utilization=0.75,
            max_model_len=4096,
            tensor_parallel_size=1,
            max_num_batched_tokens=1024,
            max_num_seqs=2,
        )
        
        # 1. Initialize Engine
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        
        web_app = FastAPI()
        
        # Middleware to intercept and modify requests
        @web_app.middleware("http")
        async def limit_max_tokens(request: Request, call_next):
            if request.url.path == "/v1/chat/completions":
                # Read the request body
                body_bytes = await request.body()
                
                if body_bytes:
                    try:
                        # Parse the JSON
                        import json
                        data = json.loads(body_bytes)
                        
                        # DEBUG: Print original request
                        print(f"Original request: {json.dumps(data, indent=2)}")
                        
                        # FORCEFULLY SET max_tokens to a safe value
                        if "max_tokens" in data:
                            print(f"Changing max_tokens from {data['max_tokens']} to 512")
                            data["max_tokens"] = 512
                        if "max_completion_tokens" in data:
                            print(f"Changing max_completion_tokens from {data['max_completion_tokens']} to 512")
                            data["max_completion_tokens"] = 512
                        
                        # If neither is present, add max_tokens
                        if "max_tokens" not in data and "max_completion_tokens" not in data:
                            data["max_tokens"] = 512
                            print("Added max_tokens: 512")
                        
                        # Encode back to bytes
                        modified_body = json.dumps(data).encode('utf-8')
                        
                        # Create a new request with modified body
                        from starlette.requests import Request as StarletteRequest
                        from starlette.datastructures import Headers
                        
                        # We need to create a new request with the modified body
                        # This is a workaround since request._body is read-only in newer FastAPI
                        scope = request.scope.copy()
                        scope['_body'] = modified_body
                        
                        # Create new request
                        request = Request(scope, request.receive)
                        
                        print(f"Modified request: {json.dumps(data, indent=2)}")
                        
                    except Exception as e:
                        print(f"Error in middleware: {e}")
                        import traceback
                        traceback.print_exc()
            
            response = await call_next(request)
            return response
        
        web_app.include_router(router)
        
        @web_app.get("/health")
        def health(): 
            return {
                "status": "ok",
                "gpu_memory_gb": torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0,
            }

        @web_app.get("/v1/models")
        def list_models():
            return {
                "object": "list",
                "data": [{"id": model_name, "object": "model"}]
            }

        # 2. Create Serving Models object
        base_model_paths = [BaseModelPath(name=model_name, model_path=local_model_path)]
        
        models_serving = OpenAIServingModels(
            engine_client=engine,
            base_model_paths=base_model_paths,
        )
        
        # 3. Create Serving Chat object
        chat_serving = OpenAIServingChat(
            engine,
            models_serving,
            "assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format='auto',
            trust_request_chat_template=False,
        )
        
        # 4. Final state assignment
        web_app.state.openai_serving_chat = chat_serving
        web_app.state.openai_serving_models = models_serving
        
        print(f"vLLM server ready for model: {model_name}")
        print(f"Model max context length: 4096 tokens")
        print(f"Middleware will limit max_tokens to 512")
        
        return web_app

# --- PADDLE OCR PROCESSOR ---
@app.cls(
    image=paddle_image,
    gpu="T4",
)
class PaddleOCRProcessor:
    @modal.enter()
    def start_engine(self):
        from paddleocr import PaddleOCRVL
        
        base_url = f"https://{WORKSPACE}--vllm-ocr-api.modal.run"
        vllm_url = f"{base_url}/v1"
        
        print(f"Checking vLLM availability at: {base_url}/health")
        
        max_retries = 30
        for i in range(max_retries):
            try:
                res = httpx.get(f"{base_url}/health", timeout=10.0)
                if res.status_code == 200:
                    print("vLLM is online and ready!")
                    
                    # Test the models endpoint
                    models_res = httpx.get(f"{vllm_url}/models", timeout=5.0)
                    if models_res.status_code == 200:
                        print(f"Models endpoint: {models_res.json()}")
                    
                    break
            except Exception as e:
                print(f"Retry {i+1}/{max_retries}: {e}")
            time.sleep(5)
        else:
            raise RuntimeError(f"vLLM backend failed to start.")

        self.pipeline = PaddleOCRVL(
            vl_rec_backend="vllm-server",
            vl_rec_server_url=vllm_url,
        )
        print("PaddleOCRVL pipeline initialized")

    @modal.method()
    def process_image(self, image_bytes: bytes):
        import tempfile
        import json
        from datetime import datetime
        
        # Save image temporarily
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp.write(image_bytes)
            tmp_path = tmp.name
        
        try:
            print(f"Processing image: {tmp_path}")
            
            # Run OCR
            output = self.pipeline.predict(tmp_path)
            print(f"Got {len(output)} results")
            
            # Convert to JSON-serializable format
            result_list = []
            for idx, res in enumerate(output):
                try:
                    result_dict = {}
                    
                    # Common PaddleOCR attributes
                    if hasattr(res, 'text'):
                        result_dict['text'] = res.text
                    if hasattr(res, 'confidence'):
                        result_dict['confidence'] = float(res.confidence)
                    if hasattr(res, 'coordinates'):
                        result_dict['coordinates'] = [list(coord) for coord in res.coordinates]
                    if hasattr(res, 'bbox'):
                        result_dict['bbox'] = list(res.bbox)
                    
                    # If we have no common attributes, try to convert the object
                    if not result_dict:
                        try:
                            result_dict = vars(res)
                        except:
                            result_dict = {"raw": str(res)}
                    
                    # Ensure serializable
                    clean_dict = {}
                    for key, value in result_dict.items():
                        try:
                            if hasattr(value, 'tolist'):
                                clean_dict[key] = value.tolist()
                            elif isinstance(value, (list, tuple, dict, int, float, str, bool, type(None))):
                                clean_dict[key] = value
                            else:
                                clean_dict[key] = str(value)
                        except:
                            clean_dict[key] = str(value)
                    
                    result_list.append(clean_dict)
                    
                except Exception as e:
                    print(f"Error processing result {idx}: {e}")
                    result_list.append({"error": str(e), "raw_result": str(res)})
            
            # Create final result
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            result_data = {
                "status": "success",
                "results": result_list,
                "count": len(result_list),
                "timestamp": timestamp
            }
            
            # Print to console
            print("OCR Results Summary:")
            print(f"  - Total items: {len(result_list)}")
            for i, item in enumerate(result_list[:3]):  # Show first 3
                if 'text' in item:
                    print(f"  - Item {i}: {item['text'][:50]}...")
            
            return result_data
            
        except Exception as e:
            print(f"Error in OCR processing: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": str(e)}
            
        finally:
            # Cleanup
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# --- MAIN GATEWAY ---
@app.function(image=paddle_image)
@modal.asgi_app()
def fastapi_app():
    from fastapi import FastAPI, UploadFile, File, HTTPException
    from fastapi.responses import JSONResponse
    
    web_app = FastAPI(title="PaddleOCR-VL Serverless API")

    @web_app.post("/ocr")
    async def run_ocr(file: UploadFile = File(...)):
        content = await file.read()
        try:
            processor = PaddleOCRProcessor()
            result = processor.process_image.remote(content)
            return JSONResponse(result)
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @web_app.get("/")
    def root():
        return {"message": "PaddleOCR-VL API", "endpoints": ["POST /ocr"]}

    return web_app