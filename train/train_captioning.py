# Path: Vision.ai/train_captioning.py

# Setup: import project root so "src" can be imported properly
import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm # For progress bar

# 🧠 Add project root directory to sys.path
# This allows importing modules from the 'src' directory using 'src.captioning.module_name'
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
# print(f"Added {project_root} to sys.path") # Optional: for debugging path issues

# STEP 1: Import necessary modules from your src directory and PyTorch
# Ensure your src/captioning/__init__.py exists (can be empty) for these imports to work
# Import the actual collate_fn under an alias
from src.captioning.dataset import CaptionDataset, collate_fn as actual_collate_fn
from src.captioning.vocabulary import Vocabulary
from src.captioning.utils import load_captions # Assuming load_captions from utils is used
from src.captioning.transforms import get_transform # Assuming get_transform from transforms is used
from src.captioning.encoder_decoder import CaptioningModel # Assuming CaptioningModel from encoder_decoder is used

# Optional: NLTK download if Vocabulary or utils require it and it's not pre-downloaded
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    print("NLTK punkt tokenizer not found, downloading...")
    nltk.download('punkt')
except LookupError:
     print("NLTK punkt tokenizer not found, attempting download...")
     nltk.download('punkt')


# --- Define Class-based Collate (OUTSIDE if __name__ == "__main__":) ---
# This class must be defined at the top level of the module to be picklable.
class TrainCollate:
    """
    A picklable class-based collate function for DataLoader workers.
    Stores the padding value and calls the actual collate_fn from the dataset module.
    An instance of this class is passed to the DataLoader's collate_fn parameter.
    """
    def __init__(self, padding_value):
        # Store the padding value as an instance attribute
        self.padding_value = padding_value

    def __call__(self, batch):
        """
        This method makes an instance of this class callable like a function.
        It is called by the DataLoader worker to process a batch.
        """
        # Call the actual collate_fn imported from dataset.py, passing the stored padding value
        return actual_collate_fn(batch, self.padding_value) # Pass the stored padding_value

# --- End of Class-based Collate ---


# STEP 2: Set paths for your data files
# These paths are resolved relative to the project root directory
# Update these paths if your data is located elsewhere
# Based on your output, your train script is in a 'train' subdir, so '..' goes up one level to Vision.ai
project_root_from_train = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Ensure paths are correct relative to where THIS script is run.
# The logic sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# correctly adds the project root.
# Data paths should also be relative to this project root.
image_folder = os.path.join(project_root_from_train, "Data/images/Flicker8k_Dataset")
caption_file = os.path.join(project_root_from_train, "Data/captions/Flickr8k_text/Flickr8k.token.txt")

# Check if data paths exist
if not os.path.exists(image_folder):
    print(f"Error: Image folder not found at {image_folder}")
    sys.exit(1) # Exit if data is missing
if not os.path.exists(caption_file):
    print(f"Error: Caption file not found at {caption_file}")
    sys.exit(1) # Exit if data is missing


# STEP 3: Prepare dataset and vocabulary
# Load and clean captions using the corrected load_captions from utils.py
caption_dict = load_captions(caption_file)

# Check if any captions were loaded
if not caption_dict:
    print("Error: No captions loaded or all were filtered out. Cannot build vocabulary or dataset.")
    sys.exit(1)


# Initialize vocabulary and build it from the loaded captions
vocab = Vocabulary(freq_threshold=5) # Using a frequency threshold
vocab.build_vocabulary(caption_dict) # Builds vocab based on cleaned captions

# Check if vocabulary was built successfully (more than just special tokens)
if len(vocab) <= 4:
     print("Error: Vocabulary contains only special tokens. Check data and frequency threshold.")
     sys.exit(1)


# Get the image transformation pipeline
# Consider using get_train_transform() if you add augmentation
transform = get_transform() # Using the basic transform for now

# Create the dataset instance
dataset = CaptionDataset(
    image_folder=image_folder,
    caption_dict=caption_dict, # caption_dict now contains cleaned captions
    vocabulary=vocab,
    transform=transform
)

# Check if dataset is empty after filtering missing images/empty captions
if len(dataset) == 0:
     print("Error: Dataset is empty after checking image files and cleaning captions. Cannot train.")
     sys.exit(1)
else:
    print(f"Successfully created dataset with {len(dataset)} image-caption pairs.")


# --- Setup for DataLoader and Collate ---
# Get the padding index from the vocabulary *after* it's built
pad_idx = vocab.stoi["<pad>"]
print(f"Padding token index: {pad_idx}")

# Instantiate the picklable collate class, passing the padding value to its constructor
# This instance will be passed to the DataLoader
my_collate_fn_instance = TrainCollate(padding_value=pad_idx)

# Create the DataLoader for batching data during training
data_loader = DataLoader(
    dataset,
    batch_size=32, # Adjust batch size based on your system's memory
    shuffle=True,  # Shuffle data for better training
    num_workers=2, # num_workers > 0 is now okay with the picklable class instance
    collate_fn=my_collate_fn_instance # Pass the *instance* of the collate class
)
# --- End of DataLoader Setup ---


# STEP 4: Initialize model and training tools
# Define model parameters - ensure these match what you intend for your CaptioningModel
vocab_size = len(vocab) # Vocabulary size based on the built vocabulary
embed_size = 512      # Dimension of word embeddings and model features (d_model)
num_heads = 8         # Number of attention heads in Transformer
hidden_dim = 2048     # Dimension of feedforward network in Transformer
num_layers = 3        # Number of Transformer decoder layers
dropout = 0.1         # Dropout rate

# --- Fine-tuning Configuration ---
# Decide which encoder layers to fine-tune
# Set to None or [] to keep the entire encoder frozen (original behavior)
# ResNet layers: 'layer1', 'layer2', 'layer3', 'layer4'
# Fine-tuning later layers is common. Start small, e.g., just 'layer4'.
fine_tune_encoder_layers = ['layer4'] # Example: fine-tune only the last ResNet layer
# fine_tune_encoder_layers = ['layer3', 'layer4'] # Example: fine-tune last two layers
# fine_tune_encoder_layers = None # Example: keep encoder frozen


# Set up device for training (GPU if available, otherwise CPU or MPS)
# MPS is for Apple Silicon GPUs
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"✅ Using {device} for training")

# Initialize the CaptioningModel and move it to the selected device
# Pass the configuration for fine-tuning to the model constructor
model = CaptioningModel(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    max_len=100, # Use a reasonable max_len for Positional Encoding. Consider max caption length + 2.
    fine_tune_encoder_layers=fine_tune_encoder_layers # Pass the fine-tuning config
).to(device)

# Define the loss function - Cross-Entropy Loss is standard for classification over vocabulary
# ignore_index=pad_idx ensures that the loss is not calculated for padding tokens
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)

# Define the optimizer - Adam is a common choice
# If fine-tuning, you are training more parameters.
# It's often best to use a smaller learning rate for the fine-tuned layers than for the new layers (fc, bn) and the decoder.
# A simple approach is to use a slightly smaller learning rate overall if fine-tuning.
# A better approach is to use parameter groups:

# Get trainable parameters
trainable_params = filter(lambda p: p.requires_grad, model.parameters())

# Simple Optimizer (all trainable params get the same LR)
# learning_rate = 3e-4 # Original LR
# if fine_tune_encoder_layers:
#      learning_rate = 5e-5 # Example: Use a smaller LR when fine-tuning

# optimizer = optim.Adam(trainable_params, lr=learning_rate)


# --- Optimizer with Parameter Groups (Recommended for Fine-tuning) ---
# This allows setting different learning rates for different parts of the model.
# Parameters that require grad will be split into encoder (fine-tuned layers) and others (decoder, new fc/bn).
encoder_params = []
other_params = [] # Includes decoder, fc, bn

for name, param in model.named_parameters():
    if param.requires_grad: # Only consider parameters that are being trained
        # Check if the parameter name is part of the fine-tuned encoder layers
        # The names will look like 'encoder.layer4.0.conv1.weight' etc.
        if fine_tune_encoder_layers and any(layer_name in name for layer_name in fine_tune_encoder_layers):
             encoder_params.append(param)
             # print(f"Adding encoder param to group: {name}") # Debugging
        else:
             other_params.append(param)
             # print(f"Adding other param to group: {name}") # Debugging


# Check if both lists have parameters if fine-tuning is enabled
if fine_tune_encoder_layers and (not encoder_params or not other_params):
    print("Warning: Fine-tuning is enabled, but parameter groups might not be correctly split.")
    print(f"Encoder params group size: {len(encoder_params)}")
    print(f"Other params group size: {len(other_params)}")
    # Fallback to simple optimizer if parameter groups seem wrong
    print("Falling back to optimizing all trainable params with the same LR.")
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=3e-4)
else:
    # Define learning rates
    # Use a much smaller LR for the fine-tuned encoder layers
    encoder_lr = 1e-5 if fine_tune_encoder_layers else 3e-4 # Smaller LR if fine-tuning, else main LR (though this group should be empty if not fine-tuning)
    other_lr = 3e-4 # Main LR for decoder and new layers

    # Create optimizer with parameter groups
    optimizer = optim.Adam([
        {'params': encoder_params, 'lr': encoder_lr}, # Group 0: Encoder fine-tuned layers (if any)
        {'params': other_params, 'lr': other_lr}       # Group 1: Decoder, fc, bn
    ], lr=other_lr) # The main 'lr' in Adam constructor is a default if not specified in groups, but set it to other_lr explicitly


print(f"Optimizer set up. Training parameters: {len(list(model.parameters()))}. Trainable parameters: {len(list(filter(lambda p: p.requires_grad, model.parameters())))}")
if fine_tune_encoder_layers:
     print(f"Fine-tuning layers: {fine_tune_encoder_layers}. Encoder group LR: {optimizer.param_groups[0]['lr']}, Other group LR: {optimizer.param_groups[1]['lr']}")


# STEP 5: Training Loop
num_epochs = 10 # Define the number of training epochs (you can change this)

# Ensure the script runs the training loop when executed directly
if __name__ == "__main__":
    print(f"\n🚀 Starting training for {num_epochs} epochs...")

    # Add a try-except block for potentially cleaner exit on errors
    try:
        for epoch in range(num_epochs):
            # Set the model to training mode (enables dropout, batchnorm updates)
            model.train()
            running_loss = 0.0 # To track loss over the epoch

            # Use tqdm for a progress bar over the DataLoader
            # Use total=len(data_loader) for tqdm to estimate time
            loop = tqdm(data_loader, leave=True, total=len(data_loader))

            # Iterate over batches from the data_loader
            for batch_idx, (images, captions) in enumerate(loop):
                # Ensure batch contains data (collate_fn might return None, None if empty)
                if images is None or captions is None:
                     # print(f"Skipping empty batch at epoch {epoch+1}, batch {batch_idx}")
                     continue

                # Move the data batch to the selected device (GPU/CPU/MPS)
                images, captions = images.to(device), captions.to(device)

                # Prepare the input and target for the decoder using Teacher Forcing
                # tgt_input: Shifted right (e.g., <sos> word1 word2 ... wordN)
                # tgt_output: Original sequence (e.g., word1 word2 word3 ... wordN <eos>)
                # We use[:, :-1] for input sequence (remove <eos>)
                # We use[:, 1:] for target sequence (remove <sos>)
                # Ensure captions are long tensors as expected by embedding layer and loss
                tgt_input = captions[:, :-1].long()
                tgt_output = captions[:, 1:].long()

                # Create the causal mask for the target sequence
                # This mask prevents the decoder from attending to future tokens
                # Size is based on the sequence length of the target input (tgt_input)
                # nn.Transformer.generate_square_subsequent_mask creates a boolean mask where True is *masked*
                # Its size is (sequence_length, sequence_length)
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
                # If using PyTorch < 1.9, the mask might need to be float and have -inf where True
                # Check PyTorch docs for nn.Transformer.generate_square_subsequent_mask for your version.
                # For newer versions, boolean mask is often handled internally.


                # Perform the forward pass through the model
                # model takes images, target input sequence, and the causal mask
                # output shape: [B, T-1, vocab_size]
                outputs = model(images, tgt_input, tgt_mask)

                # Reshape outputs and targets for the CrossEntropyLoss calculation
                # CrossEntropyLoss expects input shape [N, C] and target shape [N]
                # N = Batch Size * Sequence Length (which is B * (T-1) here)
                # C = Vocab Size
                outputs = outputs.reshape(-1, outputs.shape[2]) # Reshape from [B, T-1, vocab_size] to [(B*(T-1)), vocab_size]
                tgt_output = tgt_output.reshape(-1)           # Reshape from [B, T-1] to [(B*(T-1))]

                # Calculate the loss
                loss = criterion(outputs, tgt_output)

                # Perform backpropagation and update model weights
                optimizer.zero_grad() # Clear previous gradients
                loss.backward()       # Compute gradients
                optimizer.step()      # Update weights

                running_loss += loss.item() # Use .item() to get scalar value for accumulation
                loop.set_postfix(loss=loss.item()) # Display current batch loss in tqdm

            # Print average loss for the epoch
            avg_epoch_loss = running_loss / len(data_loader)
            print(f"\n✅ Epoch {epoch+1} completed — Average Loss: {avg_epoch_loss:.4f}")

    except Exception as e:
        print(f"\n🚫 An error occurred during training: {e}")
        import traceback
        traceback.print_exc() # Print the full traceback
        sys.exit(1) # Exit if training failed


    print("\n✨ Training finished.")

    # ✅ Save model state dictionary and vocabulary after training
    # Define paths for saving relative to the script location
    model_save_path = os.path.join(project_root_from_train, "model.pth")
    vocab_save_path = os.path.join(project_root_from_train, "vocab.pth")

    try:
        # Save the model's learned state dictionary (weights and biases)
        torch.save(model.state_dict(), model_save_path)

        # Save the vocabulary object.
        # Saving the entire object works but requires torch.serialization.add_safe_globals
        # in the loading script and is less secure. Saving/loading state is safer.
        torch.save(vocab, vocab_save_path)
        # If you modified Vocabulary to have get_state/from_state:
        # torch.save(vocab.get_state(), os.path.join(project_root_from_train, "vocab_state.pth"))

        print(f"📦 Model state dictionary saved to {model_save_path}")
        print(f"📦 Vocabulary object saved to {vocab_save_path}")

    except Exception as e:
        print(f"🚫 Error saving model or vocabulary: {e}")
        import traceback
        traceback.print_exc()