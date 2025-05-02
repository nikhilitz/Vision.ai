# Path: Vision.ai/src/captioning/dataset.py

import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence # Ensure this import is present

# Assuming Vocabulary class is imported correctly from the same package
# from .vocabulary import Vocabulary # Example relative import within src.captioning
# Assuming Vocabulary is imported directly in the script using this class (like train_captioning.py)
# from src.captioning.vocabulary import Vocabulary

# Assuming utils.load_captions and utils.clean_caption are used to prepare caption_dict
# from .utils import load_captions, clean_caption # Example relative import

class CaptionDataset(Dataset):
    def __init__(self, image_folder, caption_dict, vocabulary, transform=None):
        """
        Initializes the dataset by checking image file existence and creating
        a list of valid image-caption pairs.

        :param image_folder: Path to image directory
        :param caption_dict: Dictionary {image_id: [list of cleaned captions]} (cleaned using utils.clean_caption)
        :param vocabulary: Vocabulary object
        :param transform: torchvision transforms to apply to image
        """
        self.image_folder = image_folder
        self.vocab = vocabulary
        self.transform = transform

        # Flatten image-caption pairs into a list for easy indexing
        # We will only add pairs where the image file exists and has non-empty cleaned captions
        self.image_caption_pairs = []
        skipped_images_count = 0
        skipped_caption_entries = 0 # Captions skipped due to missing image or being empty after cleaning
        # Assuming caption_dict values are lists of *cleaned* captions now due to utils.py change
        total_caption_entries_initial = sum(len(caps) for caps in caption_dict.values())


        print("🔎 Checking image file existence and building dataset...")

        # Iterate through image IDs from the caption dictionary
        # caption_dict keys are image_ids (e.g., '1000268201_693b08cb0e.jpg')
        valid_image_ids_count = 0
        processed_caption_count = 0

        for image_id, captions in caption_dict.items():
            # Ensure image_id is a non-empty string
            if not isinstance(image_id, str) or not image_id:
                 skipped_images_count += 1
                 # Count how many captions were associated with this invalid key
                 if isinstance(captions, list):
                      skipped_caption_entries += len(captions)
                 continue # Skip this entry if image_id is invalid

            image_path = os.path.join(self.image_folder, image_id)

            # Check if the corresponding image file exists on disk
            if os.path.exists(image_path):
                 valid_image_ids_count += 1
                 # If the image exists, add all its *non-empty* captions as separate entries
                 # Captions should already be cleaned by load_captions
                 if isinstance(captions, list):
                     valid_captions = [cap for cap in captions if isinstance(cap, str) and cap] # Ensure it's a non-empty string

                     if valid_captions:
                         for caption in valid_captions:
                             self.image_caption_pairs.append((image_id, caption))
                             processed_caption_count += 1
                     else:
                         # If the image exists but has no valid captions after cleaning
                         skipped_caption_entries += len(captions) # Count original number of captions for this image
                 else:
                      # If captions value was not a list
                      skipped_caption_entries += 0 # Or log this unusual case


            else:
                # If the image file does not exist, skip this image and its captions
                skipped_images_count += 1
                if isinstance(captions, list):
                     skipped_caption_entries += len(captions)


        print(f"\n--- Dataset Loading Summary ---")
        print(f"Total unique image IDs from captions dictionary: {len(caption_dict)}")
        print(f"Total caption entries from captions dictionary (before image/cleaning check): {total_caption_entries_initial}")
        print(f"Image files found on disk: {valid_image_ids_count}")
        print(f"Image files not found (and skipped): {skipped_images_count}")
        print(f"Caption entries skipped (due to missing images or becoming empty after cleaning): {skipped_caption_entries}")
        print(f"Dataset size (valid image-caption pairs): {len(self.image_caption_pairs)}") # This is the count of individual (image, caption) pairs
        print(f"-------------------------------\n")


    def __len__(self):
        # The length is the number of valid image-caption pairs found
        return len(self.image_caption_pairs)

    def __getitem__(self, index):
        # This method will now only be called for image_id's that were confirmed to exist and had valid captions in __init__
        image_id, caption = self.image_caption_pairs[index]
        image_path = os.path.join(self.image_folder, image_id)

        # Open the image file
        # We assume image_path exists because we pre-filtered in __init__
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
             print(f"Error loading image {image_path} at __getitem__: {e}")
             # This should ideally not happen with the __init__ check, but is a safeguard.
             # Depending on requirements, you might return None or a dummy item,
             # but re-raising might be better to catch unexpected issues.
             raise # Re-raise the exception if loading fails

        if self.transform:
            image = self.transform(image)

        # numericalize caption (assuming caption is already cleaned string by load_captions)
        # Add SOS and EOS tokens
        numericalized_caption = [self.vocab.stoi['<sos>']]
        # vocab.numericalize returns a list of token indices from a string
        # This implicitly uses the vocabulary's tokenizer (which expects cleaned text)
        numericalized_caption.extend(self.vocab.numericalize(caption))
        numericalized_caption.append(self.vocab.stoi['<eos>'])

        # Return as a PyTorch tensor
        return image, torch.tensor(numericalized_caption, dtype=torch.long) # Specify dtype

# The collate_fn is defined outside the class
# It should accept the padding_value from the DataLoader
def collate_fn(batch, padding_value): # Added padding_value parameter, removed default
    """
    Pads caption sequences with a specified value and stacks images for a batch.
    This function is intended to be passed to the DataLoader.
    :param batch: A list of (image, numericalized_caption_tensor) tuples.
    :param padding_value: The index to use for padding (must match vocab.<pad> index).
    """
    # Ensure batch is not empty
    if not batch:
        # print("Warning: Received empty batch in collate_fn")
        return None, None # Or handle as appropriate for your DataLoader setup

    # Separate images and captions from the batch
    # zip(*) will handle if images or captions are missing in a batch (though __init__ and __getitem__ should prevent this)
    try:
        images, captions = zip(*batch)
    except ValueError as e:
        print(f"Error during zip(*batch) in collate_fn: {e}. Batch content might be inconsistent.")
        # Inspect batch content if this happens
        # print("Batch content sample:", batch[:5]) # Print first few items
        raise # Re-raise to stop execution and debug

    # Stack images into a single batch tensor
    # Assumes images are already resized and are Tensors of shape [C, H, W]
    # Ensure all images in the batch are valid tensors before stacking
    valid_images = [img for img in images if isinstance(img, torch.Tensor)]
    if not valid_images:
         # print("Warning: No valid image tensors in batch for stacking.")
         return None, None # Or handle appropriately

    images = torch.stack(valid_images, dim=0) # Result shape [B, C, H, W]

    # Pad sequences in the captions tuple to the maximum length in the batch
    # pad_sequence expects a list or tuple of tensors
    # batch_first=True puts the batch dimension first: [B, T]
    # padding_value is the index used for padding
    # Ensure captions are tensors before padding
    valid_captions = [cap for cap in captions if isinstance(cap, torch.Tensor)]
    if not valid_captions:
         # print("Warning: No valid caption tensors in batch for padding.")
         return images, None # Return images, but no captions


    # Check if padding_value is a valid integer index
    if not isinstance(padding_value, int):
         print(f"Error: Invalid padding_value type in collate_fn: {type(padding_value)}. Expected int.")
         # Fallback to 0 or raise error
         padding_value = 0 # Fallback
         # raise TypeError(f"Invalid padding_value type: {type(padding_value)}. Expected int.")


    try:
        # All caption tensors in valid_captions should have dtype=torch.long from __getitem__
        captions = pad_sequence(valid_captions, batch_first=True, padding_value=padding_value) # Result shape [B, max_T_in_batch]
        return images, captions
    except Exception as e:
        print(f"Error during pad_sequence in collate_fn: {e}")
        # Inspect valid_captions content/dtypes if this happens
        # print("Valid captions sample shapes/dtypes:", [(c.shape, c.dtype) for c in valid_captions[:5]])
        raise # Re-raise