import httpx
import asyncio
import sys

# The URL from your deployment output + the endpoint path
URL = "https://info-nchat--paddleocr-vl-vllm-deployment-fastapi-app.modal.run/ocr"

async def perform_ocr(image_path: str):
    print(f"Sending {image_path} to Modal...")

    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            with open(image_path, "rb") as f:
                # 'file' must match the parameter name in the FastAPI run_ocr function
                files = {"file": (image_path, f, "image/png")}
                response = await client.post(URL, files=files)

            if response.status_code == 200:
                print("OCR Result Received:")
                print(response.json())
            else:
                print(f"Error {response.status_code}: {response.text}")

        except Exception as e:
            print(f"Request failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_image>")
    else:
        asyncio.run(perform_ocr(sys.argv[1]))