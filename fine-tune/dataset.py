import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path


class NepaliOCRDataset(Dataset):
    """
    Dataset for Nepali handwritten OCR.
    CSV format: image_path,text
    """

    def __init__(self, csv_path, data_dir, processor=None, max_image_size=None):
        """
        Args:
            csv_path: Path to CSV file with image_path and text columns
            data_dir: Root directory containing 'cropped' folder with images
            processor: HuggingFace processor for the model
            max_image_size: Optional max dimension for images (None = no resizing)
        """
        self.data_dir = Path(data_dir)
        self.df = pd.read_csv(csv_path)
        self.processor = processor
        self.max_image_size = max_image_size

        # Validate paths exist
        missing = []
        for idx, row in self.df.iterrows():
            img_path = self.data_dir / "cropped" / row['image_path']
            if not img_path.exists():
                missing.append(row['image_path'])

        if missing:
            print(f"Warning: {len(missing)} images not found")
            self.df = self.df[~self.df['image_path'].isin(
                missing)].reset_index(drop=True)

        print(f"Loaded {len(self.df)} valid samples from {csv_path}")
        if self.max_image_size:
            print(
                f"Max image size set to: {self.max_image_size}x{self.max_image_size}")

    def __len__(self):
        return len(self.df)

    def _resize_image(self, image):
        """Resize image if it exceeds max_image_size while maintaining aspect ratio."""
        if self.max_image_size is None:
            return image

        width, height = image.size
        max_dim = max(width, height)

        if max_dim > self.max_image_size:
            scale = self.max_image_size / max_dim
            new_width = int(width * scale)
            new_height = int(height * scale)
            image = image.resize((new_width, new_height),
                                 Image.Resampling.LANCZOS)

        return image

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.data_dir / "cropped" / row['image_path']
        image = Image.open(img_path).convert("RGB")
        image = self._resize_image(image)
        text = str(row['text']).strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "OCR:"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": text}],
            },
        ]

        return {
            "image": image,
            "text": text,
            "messages": messages,
            "image_path": str(row['image_path'])
        }


def collate_fn_minimal_masking(examples, processor, pad_to_multiple_of=8):
    """
    Alternative collate function with MINIMAL masking.
    Only masks the image tokens and special tokens, keeps most text.

    This is useful if you want the model to learn the entire conversation,
    including "OCR:" prompt.
    """
    images = []
    messages = []

    for example in examples:
        images.append(example["image"])
        messages.append(example["messages"])

    texts = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    batch = processor(
        text=texts,
        images=images,
        padding=True,
        pad_to_multiple_of=pad_to_multiple_of,
        return_tensors="pt",
        add_special_tokens=False,
    )

    labels = batch["input_ids"].clone()

    # Only mask padding tokens
    labels[batch["attention_mask"] == 0] = -100

    # Optionally: Mask special tokens like <|begin_of_sentence|>
    special_token_ids = [
        processor.tokenizer.bos_token_id,
        processor.tokenizer.eos_token_id,
    ]

    for special_id in special_token_ids:
        if special_id is not None:
            labels[labels == special_id] = -100

    batch["labels"] = labels

    # Debug
    if not hasattr(collate_fn_minimal_masking, '_debug_printed'):
        non_masked = (labels[0] != -100).sum().item()
        print(
            f"\nMINIMAL MASKING mode: {non_masked} learning tokens (most of sequence)")
        collate_fn_minimal_masking._debug_printed = True

    return batch
