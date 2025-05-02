# Path: Vision.ai/src/captioning/utils.py

import re
from nltk.translate.bleu_score import sentence_bleu
import nltk

# Ensure 'punkt' tokenizer data is available for NLTK
# nltk.download('punkt') # Often handled once during setup or in init files

def clean_caption(caption):
    """
    Cleans caption text by:
    1. Lowercasing
    2. Removing punctuation (keeping only letters and spaces)
    3. Removing extra spaces
    Handles non-string inputs gracefully.
    """
    if not isinstance(caption, str):
        # print(f"Warning: clean_caption received non-string input: {type(caption)}")
        return ""

    caption = caption.lower()
    # Regex keeps letters (a-z) and spaces (\s). Removes everything else.
    # Removed 0-9 based on typical caption content - adjust if numbers are important
    caption = re.sub(r"[^a-z\s]", "", caption)
    caption = re.sub(r"\s+", " ", caption) # Replace multiple spaces with a single space
    return caption.strip() # Remove leading/trailing spaces

def load_captions(file_path):
    """
    Loads captions from a file, cleans them using clean_caption, and organizes by image ID.
    Assumes file format is image_id#index \\t caption_text per line.
    Handles potential empty lines, missing tabs, and empty captions after cleaning.
    """
    caption_dict = {}
    print(f"Loading and cleaning captions from {file_path}...")
    skipped_lines_count = 0
    skipped_empty_captions_count = 0
    total_lines_read = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                total_lines_read += 1
                line = line.strip()
                # Skip empty lines
                if not line:
                    skipped_lines_count += 1
                    continue

                # Ensure the line has a tab separator
                if '\t' not in line:
                    # print(f"Warning: Skipping line with no tab separator: '{line}'")
                    skipped_lines_count += 1
                    continue

                try:
                    # Split only on the first tab to handle tabs potentially within captions
                    parts = line.split('\t', 1)
                    if len(parts) < 2: # Should not happen with '\t' check, but defensive
                        skipped_lines_count += 1
                        continue

                    image_part, caption_text = parts

                    # Extract image ID from image_part (e.g., "1000268201_693b08cb0e.jpg#0")
                    image_id_parts = image_part.split('#')
                    image_id = image_id_parts[0].strip() # Take the part before #

                    # Apply the consistent cleaning function to the caption text
                    cleaned_caption = clean_caption(caption_text)

                    # Only include the caption if it's not empty after cleaning
                    if cleaned_caption:
                        if image_id not in caption_dict:
                            caption_dict[image_id] = []
                        caption_dict[image_id].append(cleaned_caption)
                    else:
                        skipped_empty_captions_count += 1
                        # Optional: print(f"Skipped empty caption after cleaning for image part '{image_part}': '{caption_text}'")

                except Exception as e:
                    # Catch potential errors during parsing or cleaning a line
                    # print(f"Warning: Skipping problematic line due to error: {e}\nLine content: '{line}'")
                    skipped_lines_count += 1
                    continue

    except FileNotFoundError:
        print(f"Error: Caption file not found at {file_path}")
        return {} # Return empty dictionary or handle error as appropriate
    except Exception as e:
        print(f"An error occurred while reading the caption file {file_path}: {e}")
        return {} # Return empty dictionary or handle error as appropriate


    total_loaded_captions = sum(len(caps) for caps in caption_dict.values())
    print(f"Finished loading {total_loaded_captions} cleaned captions for {len(caption_dict)} unique images.")
    if skipped_lines_count > 0 or skipped_empty_captions_count > 0:
         print(f"Skipped {skipped_lines_count} lines (format issues) and {skipped_empty_captions_count} captions (became empty after cleaning).")
    print("-" * 30)

    return caption_dict

def get_bleu_score(reference_captions, generated_caption_tokens):
    """
    Calculates BLEU score of a generated caption against multiple reference captions.
    Applies cleaning to reference captions before splitting into tokens for consistency.
    Assumes reference_captions is a list of original (or consistently formatted) strings.
    Assumes generated_caption_tokens is the generated caption as a list of tokens (strings).

    Note: For meaningful BLEU, generated_caption_tokens should ideally not include
    special tokens like '<sos>', '<eos>', '<pad>'. Filter these out before calling this function.
    """
    if not reference_captions or not isinstance(reference_captions, list):
        # print("Warning: No reference captions provided or invalid format, BLEU score will be 0.")
        return 0.0

    if not generated_caption_tokens or not isinstance(generated_caption_tokens, list):
        # print("Warning: Generated caption is empty or invalid format, BLEU score will be 0.")
        return 0.0

    # Prepare the reference captions: clean each string and split into tokens
    # BLEU score references should be a list of lists of tokens
    references_for_bleu = []
    for ref in reference_captions:
        if isinstance(ref, str):
            cleaned_ref_tokens = clean_caption(ref).split()
            if cleaned_ref_tokens: # Only add if cleaning/splitting didn't result in empty list
                 references_for_bleu.append(cleaned_ref_tokens)


    # Filter out any empty reference lists that might result from cleaning/splitting
    # (already handled in the loop above)
    # references_for_bleu = [ref_list for ref_list in references_for_bleu if ref_list] # Redundant check

    # If no valid references remain after cleaning, return 0
    if not references_for_bleu:
        # print("Warning: All reference captions became empty after cleaning, BLEU score will be 0.")
        return 0.0

    # The hypothesis is the generated caption, expected as a list of tokens (strings)
    hypothesis = generated_caption_tokens # Assumed to be list of strings (tokens)

    # Calculate the BLEU score
    # sentence_bleu(references, hypothesis, weights)
    # weights default to (0.25, 0.25, 0.25, 0.25) for BLEU-4

    try:
         # Convert hypothesis tokens to lowercase for case-insensitivity, consistent with cleaning
         hypothesis_lower = [token.lower() for token in hypothesis if isinstance(token, str)]
         # Ensure hypothesis is not empty after lowercasing/filtering
         if not hypothesis_lower:
              # print("Warning: Generated caption became empty after processing, BLEU score will be 0.")
              return 0.0

         return sentence_bleu(references_for_bleu, hypothesis_lower)
    except Exception as e:
         # print(f"Error calculating BLEU score: {e}")
         # This might happen if inputs are not in expected list of strings/list of list of strings format
         return 0.0