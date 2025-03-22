# test/test_transforms.py

import os
import sys
from PIL import Image

# Make project root accessible
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.append(PROJECT_ROOT)

from src.captioning.transforms import get_transform

# ---------- TEST: Image Transform ----------
print("🔹 TEST: Image Transformation Pipeline")

# Define the sample image path
sample_image_path = os.path.join(PROJECT_ROOT, "Data/images/Flicker8k_Dataset/667626_18933d713e.jpg")  # Adjust as per your folder structure

# Check if the sample image exists
if os.path.exists(sample_image_path):
    try:
        transform = get_transform()
        image = Image.open(sample_image_path).convert("RGB")
        image_tensor = transform(image)

        print("✅ Image transformation successful.")
        print("📐 Transformed Image Tensor Shape:", image_tensor.shape)
        print("🔢 Tensor Values (Sample):", image_tensor[:, :2, :2])  # print a small part of the tensor
    except Exception as e:
        print("❌ Error during transformation:", e)
else:
    print(f"⚠️ Sample image not found at: {sample_image_path}")
