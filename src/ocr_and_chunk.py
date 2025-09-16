import pytesseract
from PIL import Image
from pathlib import Path
import json

# Parameters
tiff_dir = Path("/Users/shivkpatel/Desktop/data/idl_data/extracted")
output_json = "ocr_chunks.json"
chunk_size = 1000
overlap = 200

def chunk_text(text, chunk_size=1000, overlap=200):
    chunks = []
    start = 0
    text_length = len(text)
    while start < text_length:
        end = min(start + chunk_size, text_length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == text_length:
            break
        start += chunk_size - overlap
    return chunks

# Gather all .tif files
tiff_files = sorted(tiff_dir.glob("*.tif"))

all_chunks = []
for tiff_path in tiff_files:
    print(f"OCR processing {tiff_path.name} ...")
    try:
        img = Image.open(tiff_path)
        text = pytesseract.image_to_string(img)
        chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
        for idx, chunk in enumerate(chunks):
            all_chunks.append({
                "filename": tiff_path.name,
                "chunk_id": idx,
                "chunk_text": chunk
            })
    except Exception as e:
        print(f"Error processing {tiff_path.name}: {e}")

# Save chunks to JSON
with open(output_json, "w") as f:
    json.dump(all_chunks, f, indent=2)

print(f"Done! {len(all_chunks)} chunks saved to {output_json}")
