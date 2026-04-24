import time
import modal
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import ORJSONResponse
from pydantic import BaseModel
from typing import List
from modal_app import app, image, PaddleOCRService


# ============================================================================
# MODELS
# ============================================================================

class ImageRequest(BaseModel):
    image_base64: str
    prompt: str = "ocr"
    use_layout_detection: bool = True
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = True
    layout_merge_bboxes_mode: str = "small"

# ============================================================================
# FASTAPI WEB ENDPOINT
# ============================================================================


@app.function(
    image=image,
    scaledown_window=300,
    enable_memory_snapshot=False,
    timeout=900,
)
@modal.asgi_app()
def fastapi_app():
    """FastAPI web application for PaddleOCR-VL"""

    web_app = FastAPI(
        title="PaddleOCR VL API",
        description="OCR API - Scale to Zero (5 min idle)",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        default_response_class=ORJSONResponse,
    )

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    ocr_service = PaddleOCRService()

    def handle_ocr_response(result: dict, start_time: float):
        """Uniformly handles OCR service responses and errors."""
        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"OCR processing failed: {result.get('error', 'Unknown error')}"
            )

        total_latency = round((time.time() - start_time) * 1000, 2)
        return {
            **result,
            "total_latency_ms": total_latency,
            "timestamp": time.time()
        }

    @web_app.get("/")
    async def root():
        return {
            "service": "PaddleOCR VL API",
            "version": "3.0.0",
            "status": "running",
            "endpoints": ["/predict", "/health"],
        }

    @web_app.get("/health")
    async def health_check():
        return {"status": "healthy", "timestamp": time.time()}

    @web_app.post("/predict")
    async def predict(request: ImageRequest):
        """Single image OCR prediction via Base64"""
        if not request.image_base64:
            raise HTTPException(status_code=400, detail="Empty image_base64")
        
        start_time = time.time()
        result = ocr_service.predict.remote(
            image_data=request.image_base64,
            prompt=request.prompt,
            use_layout_detection=request.use_layout_detection,
            use_doc_orientation_classify=request.use_doc_orientation_classify,
            use_doc_unwarping=request.use_doc_unwarping,
            layout_merge_bboxes_mode=request.layout_merge_bboxes_mode,
            return_markdown=True,
        )
        return handle_ocr_response(result, start_time)

    return web_app
