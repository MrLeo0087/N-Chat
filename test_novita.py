import requests
import base64
import json

API_URL = "https://ccf1922cd0063a6b-8080.us-ca-6.gpu-instance.novita.ai/layout-parsing"

with open("test_image.jpg", "rb") as f:
    img_data = base64.b64encode(f.read()).decode("utf-8")

# Try to enable full OCR (may or may not work)
payload = {
    "file": img_data,
    "fileType": 1,
    "use_ocr": True,
    "return_ocr_result": True
}

response = requests.post(API_URL, json=payload)

if response.status_code != 200:
    print("API Error:", response.text)
    exit()

resp = response.json()

blocks = []
try:
    parsing_list = resp["result"]["layoutParsingResults"][0]["prunedResult"]["parsing_res_list"]
    for item in parsing_list:
        text = item.get("block_content", "").strip()
        if text:
            blocks.append({
                "text": text,
                "bbox": item["block_bbox"],
                "type": item["block_label"]
            })
except (KeyError, IndexError):
    pass

# Some deployments put line-level OCR in a different field
ocr_lines = []
try:
    # Check for nested OCR results
    layout_result = resp["result"]["layoutParsingResults"][0]
    if "ocr_result" in layout_result:
        for line in layout_result["ocr_result"].get("lines", []):
            ocr_lines.append({
                "text": line["text"],
                "bbox": line["bbox"],
                "confidence": line.get("score")
            })
except (KeyError, IndexError):
    pass

# Combine or choose
results = blocks if blocks else ocr_lines

# Save
with open("final_ocr_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"[OK] Extracted {len(results)} text elements")
if results:
    for r in results[:3]:  # show first 3
        print(f"- {r['text']} | bbox: {r['bbox']}")
else:
    print("[ERROR] No text extracted. Consider:")
    print("  1. Checking image quality")
    print("  2. Contacting Novita to enable full OCR mode")
    print("  3. Using a dedicated OCR model")