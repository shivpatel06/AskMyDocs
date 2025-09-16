import pytesseract
from PIL import Image
from pathlib import Path

# Adjust the path below if needed
tiff_dir = Path("/Users/shivkpatel/Desktop/data/idl_data/extracted")
tiff_files = list(tiff_dir.glob("*.tif"))

if not tiff_files:
    print("No TIFF files found!")
else:
    for tiff_path in tiff_files[:5]:  # Process the first 5 TIFFs
        img = Image.open(tiff_path)
        text = pytesseract.image_to_string(img)
        print(f"--- OCR from {tiff_path.name} ---")
        print(text[:500])  # Print first 500 characters
        print("\n" + "="*50 + "\n")
