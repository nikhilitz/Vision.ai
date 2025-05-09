import os
import sys
import torch
import torch.nn as nn
from PIL import Image
import nltk
from torchvision import transforms
import time  # Optional: for timing inference

# 🧠 Add project root and TTS directory to path so `src` and `TTS` can be imported properly
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'TTS'))  # Add TTS directory

# 🧠 Import your own modules AFTER setting path
# Ensure your src/captioning/__init__.py exists (can be empty) for these imports to work
from src.captioning.vocabulary import Vocabulary
from src.captioning.encoder_decoder import CaptioningModel
from src.captioning.transforms import get_transform  # Consider get_test_transform if you made one
# Assuming clean_caption from utils might be used for post-processing generated text
from src.captioning.utils import clean_caption

# Import TTS module
from tts import speak_offline  # 🔊 Import TTS module

# ✅ Allowlist custom class for torch.load (PyTorch ≥2.6)
# This is needed if you save the entire Vocabulary object instance.
# If you save/load the state (like stoi/itos dicts), this is not necessary and loading is safer.
torch.serialization.add_safe_globals([Vocabulary])

# Download punkt tokenizer if not already available locally.
# Required by NLTK word_tokenize used in Vocabulary.
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
     print("NLTK punkt tokenizer not found, downloading...")
     nltk.download("punkt")
except LookupError:
     print("NLTK punkt tokenizer not found, attempting download...")
     nltk.download("punkt")

# Device config
# Set up device for inference (GPU if available, otherwise CPU or MPS)
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"✅ Using {device} for inference")


# --- Loading Saved Model and Vocabulary ---

# Define paths for saved files
model_save_path = os.path.join(project_root, "model.pth")
vocab_save_path = os.path.join(project_root, "vocab.pth")

# Check if saved files exist
if not os.path.exists(model_save_path):
    print(f"Error: Model file not found at {model_save_path}")
    sys.exit(1)
if not os.path.exists(vocab_save_path):
    print(f"Error: Vocabulary file not found at {vocab_save_path}")
    sys.exit(1)

# 🔄 Load saved vocabulary
try:
    # Load the Vocabulary object instance directly
    vocab = torch.load(vocab_save_path, weights_only=False)
except Exception as e:
    print(f"Error loading vocabulary from {vocab_save_path}: {e}")
    sys.exit(1)

# Define model parameters - ensure these match the parameters used during training
# Get vocab_size from the loaded vocabulary
vocab_size = len(vocab)
embed_size = 512      # Must match training
num_heads = 8         # Must match training
hidden_dim = 2048     # Must match training
num_layers = 3        # Must match training
dropout = 0.1         # Must match training (though dropout layers are inactive in eval mode)
max_len = 100         # Must match training (for positional encoding)


# Initialize the CaptioningModel architecture
model = CaptioningModel(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    max_len=max_len # Use the same max_len as training
).to(device)

# Load the saved model weights onto the initialized model architecture
try:
    model.load_state_dict(torch.load(model_save_path, map_location=device))
except Exception as e:
    print(f"Error loading model state dictionary from {model_save_path}: {e}")
    sys.exit(1)

# Set the model to evaluation mode
model.eval()

print(f"✅ Model and Vocabulary loaded successfully.")


# --- Image Loading and Preprocessing ---

# Define the path to the image you want to caption
# 🔁 Change this to any image path you want to test
image_path = os.path.join(project_root, "/Users/nikhilgupta/Desktop/Deep Learning/Vision.ai/Data/images/Flicker8k_Dataset/3741827382_71e93298d0.jpg") # Example path

# Check if the image file exists
if not os.path.exists(image_path):
    print(f"Error: Input image file not found at {image_path}")
    sys.exit(1)

# Get the image transformation pipeline (must match the one used during training)
transform = get_transform()

# Load and preprocess the image
try:
    image = Image.open(image_path).convert("RGB") # Open image and ensure it's in RGB format
    image = transform(image) # Apply the transformations (resize, to tensor, normalize)
    image = image.unsqueeze(0) # Add a batch dimension at the beginning [C, H, W] -> [1, C, H, W]
    image = image.to(device) # Move the image tensor to the selected device

except Exception as e:
    print(f"Error processing image {image_path}: {e}")
    sys.exit(1)

print(f"✅ Image loaded and preprocessed from {image_path}")

# --- Caption Generation (Greedy Decoding) ---

print("\n🧠 Starting greedy decoding...")
start_time = time.time()

# Initialize the input sequence for the decoder with the Start-of-Sequence token
caption_indices = [vocab.stoi["<sos>"]] # Start with <sos> token index

# Define a maximum length for the generated caption to prevent infinite loops
max_generated_length = 30 # Generate up to 30 tokens (including <eos>)

# Loop to generate tokens one by one
for _ in range(max_generated_length):
    # Create the input tensor for the decoder from the indices generated so far
    # Add batch dimension: [seq_len] -> [1, seq_len]
    input_tensor = torch.tensor(caption_indices, dtype=torch.long).unsqueeze(0).to(device) # Specify dtype

    # Create the causal mask for the input sequence
    tgt_mask = nn.Transformer.generate_square_subsequent_mask(input_tensor.size(1)).to(device)

    # Perform a forward pass through the model to get predictions for the next token
    with torch.no_grad():
        output = model(image, input_tensor, tgt_mask)

    # Get the predictions (logits) for the *last* token in the sequence
    next_token_logits = output[0, -1]

    # Apply argmax to find the index of the token with the highest probability
    next_token_index = next_token_logits.argmax().item()

    # Append the predicted token index to the sequence
    caption_indices.append(next_token_index)

    # Check if the predicted token is the End-of-Sequence token
    if next_token_index == vocab.stoi["<eos>"]:
        break # Stop decoding if <eos> is predicted

end_time = time.time()
print(f"🧠 Greedy decoding finished in {end_time - start_time:.4f} seconds.")


# --- Convert Token Indices to Words and Print ---

# Convert the list of token indices back into a list of words
generated_tokens = [vocab.itos[idx] for idx in caption_indices if idx not in {vocab.stoi["<sos>"], vocab.stoi["<eos>"], vocab.stoi["<pad>"]}]

# Optional: Apply clean_caption again if you want to standardize the final output string
generated_string = " ".join(generated_tokens)

print(f"\n🖼️ Predicted Caption: {generated_string}")

# --- Text-to-Speech (TTS) ---

# Use the TTS module to convert the generated caption to speech
speak_offline(generated_string)
