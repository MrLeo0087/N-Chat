import requests
import base64
import sys
from pathlib import Path
import json
import mimetypes
import time
import concurrent.futures

def encode_file_to_base64(path: Path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def run_single_predict(url, file_path, output_dir):
    path = Path(file_path)
    if not path.exists():
        return {"filename": path.name, "success": False, "error": "File not found"}

    print(f"[*] Starting prediction for: {path.name}")
    
    payload = {
        "image_base64": encode_file_to_base64(path),
        "prompt": "ocr",
        "use_layout_detection": True
    }

    try:
        start_time = time.time()
        response = requests.post(
            f"{url}/predict",
            json=payload,
            timeout=300
        )
        latency = round((time.time() - start_time) * 1000, 2)

        if response.status_code != 200:
            return {
                "filename": path.name,
                "success": False,
                "error": f"HTTP {response.status_code}: {response.text}"
            }

        data = response.json()
        
        # Save JSON
        json_path = output_dir / f"{path.stem}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        # Markdown
        markdown = data.get("markdown")
        if markdown:
            (output_dir / f"{path.stem}.md").write_text(markdown, encoding="utf-8")

        # Image
        processed_image = data.get("processed_image")
        if processed_image:
            with open(output_dir / f"{path.stem}_processed.jpg", "wb") as f:
                f.write(base64.b64decode(processed_image))

        print(f"[OK] Completed {path.name} in {latency}ms")
        return {"filename": path.name, "success": True, "latency": latency}

    except Exception as e:
        return {"filename": path.name, "success": False, "error": str(e)}

def test_parallel_predict(url, file_paths, max_workers=4):
    print(f"Testing parallel /predict with {len(file_paths)} files (max_workers={max_workers})")
    
    output_dir = Path("a10-test-output")
    output_dir.mkdir(exist_ok=True)

    start_total = time.time()
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(run_single_predict, url, fp, output_dir) for fp in file_paths]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())

    total_time = round(time.time() - start_total, 2)
    print(f"\nTotal time for {len(file_paths)} images: {total_time}s")
    
    success_count = sum(1 for r in results if r["success"])
    print(f"Success: {success_count}/{len(file_paths)}")
    
    for r in results:
        if not r["success"]:
            print(f"[FAIL] {r['filename']}: {r['error']}")
            
    return success_count == len(file_paths)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python test_modal_api.py <API_URL> <FILE1> [<FILE2> ...]")
        sys.exit(1)

    api_url = sys.argv[1].rstrip("/")
    files = sys.argv[2:]

    success = test_parallel_predict(api_url, files, max_workers=len(files))
    print("\nParallel test completed:", "SUCCESS" if success else "FAILURE")
