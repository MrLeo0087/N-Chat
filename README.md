# PaddleOCR-VL Modal API

A multilingual document parsing Vision Language Model (VLM) with 0.9B parameters. Excels at recognizing tables, charts, and complex layouts.

## Local Development Setup

To create the exact virtual environment as defined in `uv.lock`:

1. **Install uv** (if needed):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Sync Dependencies**:
   ```bash
   uv sync
   ```

3. **Activate Environment**:
   ```bash
   source .venv/bin/activate
   ```

## API Endpoint
`https://info-nchat--paddleocr-vl-modal.modal.run`

## Quick Start

1. **Deploy**
   ```bash
   modal deploy api.py
   ```

2. **Test**
   ```bash
   python test_modal_api.py <URL> <IMAGE_PATH>
   ```

## Configuration Options
- `prompt`: Default `ocr`.
- `use_layout_detection`: `true`/`false`.
- `use_doc_unwarping`: `true`/`false`.
- `layout_merge_bboxes_mode`: `small` (default) or `large`.