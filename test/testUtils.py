import os
import sys

# Add the root project path dynamically to sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.captioning.utils import clean_caption, load_captions

# ---------- TEST 1: Caption Cleaning ----------
raw_caption = "A Girl is Riding, a Horse!!!"
cleaned = clean_caption(raw_caption)

print("🔹 TEST 1: Clean Caption Function")
print("Original Caption:", raw_caption)
print("Cleaned Caption :", cleaned)

# ---------- TEST 2: Load Captions ----------
caption_file_path = os.path.join(PROJECT_ROOT, "Data/captions/Flickr8k_text/Flickr8k.token.txt")

print("\n🔹 TEST 2: Load Captions Function")
if os.path.exists(caption_file_path):
    captions = load_captions(caption_file_path)
    sample_image = list(captions.keys())[0]
    print(f"Image: {sample_image}")
    for i, cap in enumerate(captions[sample_image][:3]):
        print(f"{i+1}. {cap}")
else:
    print(f"⚠️ Caption file not found at: {caption_file_path}")
