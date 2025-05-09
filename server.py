# --- server.py ---
import os
import io
import torch
import gdown
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from torchvision import transforms
from PIL import Image
import traceback # For detailed error logging

# Import transformers for translation
from transformers import MarianMTModel, MarianTokenizer, AutoTokenizer

from src.captioning.encoder_decoder import CaptioningModel
from src.captioning.vocabulary import Vocabulary
from src.captioning.transforms import get_transform
from src.captioning.utils import clean_caption

# Assume this exists and is configured.
# IMPORTANT: This function MUST support the target language text for speaking to work correctly.
# Placeholder for TTS.tts.speak_offline if not fully implemented for all languages
try:
    from TTS.tts import speak_offline
except ImportError:
    print("Warning: TTS.tts.speak_offline not found. Speaking functionality will be disabled.")
    def speak_offline(text):
        print(f"TTS (speak_offline) called with: '{text}' (TTS module not fully available or error in import)")
except Exception as e:
    print(f"Error importing TTS.tts.speak_offline: {e}. Speaking functionality will be disabled.")
    def speak_offline(text):
        print(f"TTS (speak_offline) called with: '{text}' (TTS module import failed with exception)")


# 📁 Checkpoints folder
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("translation_models", exist_ok=True) # Folder for translation models

# 📥 Download model and vocab if not already present
model_path = "checkpoints/model.pth"
vocab_path = "checkpoints/vocab.pth"

if not os.path.exists(model_path):
    print("Downloading captioning model...")
    try:
        # Ensure gdown has permissions and the link is accessible
        gdown.download("https://drive.google.com/uc?id=1BCgSWZFKGN4HtzCMGsxOSy_HtuDIly3g", model_path, quiet=False)
    except Exception as e:
        print(f"Error downloading captioning model: {e}")
        print("Please ensure gdown is installed (`pip install gdown`) and the Google Drive link is valid and public.")
        exit()

if not os.path.exists(vocab_path):
    print("Downloading vocabulary...")
    try:
        # Ensure gdown has permissions and the link is accessible
        gdown.download("https://drive.google.com/uc?id=1HxOwFzrnpj5njnvZjh6d065-7knXx2TW", vocab_path, quiet=False)
    except Exception as e:
        print(f"Error downloading vocabulary: {e}")
        print("Please ensure gdown is installed (`pip install gdown`) and the Google Drive link is valid and public.")
        exit()


# 📟 Load device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 🧠 Load Vocabulary
try:
    # Using weights_only=False is generally safer if the .pth contains more than just weights (e.g. pickled objects)
    # However, if you are certain it's just a state_dict, weights_only=True can be used for security.
    # For Vocabulary, it's likely a custom class instance, so weights_only=False is usually correct.
    vocab = torch.load(vocab_path, map_location=device, weights_only=False)
    vocab_size = len(vocab)
    print("Vocabulary loaded successfully.")
except FileNotFoundError:
    print(f"Error: Vocabulary file not found at {vocab_path}. Please ensure it's downloaded.")
    exit()
except Exception as e:
    print(f"Error loading vocabulary: {e}")
    traceback.print_exc()
    exit()


# ⚙️ Captioning Model Parameters
try:
    captioning_model = CaptioningModel(
        vocab_size=vocab_size,
        embed_size=512,
        num_heads=8,
        hidden_dim=2048,
        num_layers=3,
        dropout=0.1,
        max_len=100
    ).to(device)
except Exception as e:
    print(f"Error initializing CaptioningModel: {e}")
    traceback.print_exc()
    exit()


# 🧠 Load captioning weights
try:
    captioning_model.load_state_dict(torch.load(model_path, map_location=device))
    captioning_model.eval()
    print("Captioning model weights loaded successfully.")
except FileNotFoundError:
    print(f"Error: Captioning model file not found at {model_path}. Please ensure it's downloaded.")
    exit()
except Exception as e:
     print(f"Error loading captioning model weights: {e}")
     traceback.print_exc()
     exit()


# 🔧 Image Transform
try:
    transform = get_transform()
except Exception as e:
    print(f"Error getting transform: {e}")
    traceback.print_exc()
    exit()

# 🌍 Translation models cache
translation_models = {}
TRANSLATION_MODEL_PREFIX = 'Helsinki-NLP/opus-mt-en-'
LOCAL_TRANSLATION_CACHE_DIR = "translation_models"

def load_translation_model(target_lang_code: str):
    if target_lang_code == 'en': # No translation needed for English to English
        return None, None

    if target_lang_code in translation_models:
        print(f"Using cached translation model for {target_lang_code}.")
        return translation_models[target_lang_code]

    model_name = f'{TRANSLATION_MODEL_PREFIX}{target_lang_code}'
    print(f"Attempting to load translation model: {model_name}")
    try:
        # Ensure the cache directory exists and is writable
        os.makedirs(LOCAL_TRANSLATION_CACHE_DIR, exist_ok=True)
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=LOCAL_TRANSLATION_CACHE_DIR)
        model = MarianMTModel.from_pretrained(model_name, cache_dir=LOCAL_TRANSLATION_CACHE_DIR).to(device)
        model.eval()
        translation_models[target_lang_code] = (tokenizer, model)
        print(f"Translation model for {target_lang_code} loaded successfully.")
        return tokenizer, model
    except Exception as e: # Catching a broader exception for model loading issues
        print(f"Could not load translation model for {target_lang_code} ({model_name}): {e}")
        print(f"This might be due to an unsupported language code, network issues, or disk space problems for caching.")
        traceback.print_exc()
        if target_lang_code in translation_models: # Clean up if partially cached
             del translation_models[target_lang_code]
        return None, None

def translate_text(text: str, target_lang_code: str):
    if not text or target_lang_code == 'en': # Handle empty text or English target
        return text

    tokenizer, model = load_translation_model(target_lang_code)

    if tokenizer is None or model is None:
        return f"[Translation to {target_lang_code} failed. Model not available/loadable.] {text}"
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        with torch.no_grad():
            translated_tokens = model.generate(**inputs, num_beams=4, max_length=512, early_stopping=True)
        translated_text = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        print(f"Error during translation process to {target_lang_code}: {e}")
        traceback.print_exc()
        return f"[Translation error during text generation for {target_lang_code}: {str(e)}] {text}"


# 🚀 Initialize FastAPI app
app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Welcome to Vision.ai API"}

@app.post("/caption")
async def generate_caption(
    file: UploadFile = File(...),
    target_language: str = Form("en")
):
    # Initialize variables to ensure they are defined in all paths, especially for error responses
    english_caption = ""
    raw_translation_output = ""
    # Default to a state indicating processing hasn't successfully set these
    response_translation_status = "processing_error_occurred"
    response_translated_caption = ""
    response_translation_error = "Initialization failure or early exit."
    text_to_speak = ""

    try:
        print(f"\n--- New Caption Request ---")
        print(f"DEBUG: Received target_language: '{target_language}'")

        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        caption_indices = [vocab.stoi["<sos>"]]
        max_generation_len = 50

        captioning_model.eval()
        with torch.no_grad():
            for _ in range(max_generation_len):
                input_tensor = torch.tensor(caption_indices).unsqueeze(0).to(device)
                # Ensure your CaptioningModel's forward method matches this signature
                # Or remove tgt_mask if not used by your model's architecture
                tgt_mask = torch.nn.Transformer.generate_square_subsequent_mask(input_tensor.size(1)).to(device)
                output = captioning_model(image_tensor, input_tensor, tgt_mask)
                next_token_probs = output[0, -1]
                next_token = next_token_probs.argmax().item()
                caption_indices.append(next_token)
                if next_token == vocab.stoi["<eos>"]:
                    break
            if caption_indices[-1] != vocab.stoi["<eos>"]: # Check if loop finished due to max_len
                 print("Warning: Max caption length reached without generating <eos> token.")


        tokens = [vocab.itos[idx] for idx in caption_indices if idx not in {vocab.stoi["<sos>"], vocab.stoi["<eos>"], vocab.stoi["<pad>"]}]
        english_caption = clean_caption(" ".join(tokens))
        print(f"DEBUG: Generated english_caption: '{english_caption}'")

        # Initialize response variables based on English caption first
        response_translated_caption = english_caption # Default for UI if translation fails or not attempted
        text_to_speak = english_caption             # Default text to speak
        response_translation_status = "not_attempted" # Default status, will change if translation is tried
        response_translation_error = None           # No error initially

        if target_language and target_language.lower() != 'en':
            current_target_lang_code = target_language.lower()
            print(f"DEBUG: Attempting to translate to: '{current_target_lang_code}'")
            raw_translation_output = translate_text(english_caption, current_target_lang_code)
            print(f"DEBUG: Raw translation output from translate_text: '{raw_translation_output}'")

            if isinstance(raw_translation_output, str) and \
               (raw_translation_output.startswith("[Translation to") or raw_translation_output.startswith("[Translation error")): # More generic error check
                response_translation_status = "failed"
                split_point = raw_translation_output.find("] ") # Find end of error marker
                if split_point != -1 and raw_translation_output.startswith("["): # Ensure it's our specific format
                    response_translation_error = raw_translation_output[1:split_point] # Extract error message
                else:
                    # If format is unexpected, but indicates failure, log it
                    response_translation_error = "Translation failed with an unrecognized error format from translate_text."
                    print(f"WARNING: Unrecognized error format from translate_text: '{raw_translation_output}'")
                # On failure, response_translated_caption remains english_caption (already set)
                # text_to_speak also remains english_caption (already set)
            elif raw_translation_output and raw_translation_output.strip(): # Successfully translated and not just whitespace
                response_translation_status = "success"
                response_translated_caption = raw_translation_output
                text_to_speak = response_translated_caption # Update text_to_speak with successful translation
            else: # Empty or whitespace-only translation output, treat as failure
                response_translation_status = "failed"
                response_translation_error = "Translation resulted in empty or whitespace-only text."
                # response_translated_caption remains english_caption
                # text_to_speak remains english_caption
        elif target_language.lower() == 'en':
            response_translation_status = "not_attempted" # Correct for app.py logic when target is 'en'
            # response_translated_caption remains english_caption
            # text_to_speak remains english_caption
        else: # Should not happen with Form default, but as a fallback for unexpected target_language values
            response_translation_status = "not_attempted"
            response_translation_error = f"Invalid or unprocessed target language: {target_language}"
            print(f"WARNING: {response_translation_error}")


        print(f"DEBUG: Determined response_translation_status: '{response_translation_status}'")
        print(f"DEBUG: Determined response_translated_caption (for UI): '{response_translated_caption}'")
        print(f"DEBUG: Determined response_translation_error: '{response_translation_error}'")
        print(f"DEBUG: Text being sent to speak_offline: '{text_to_speak}'")

        # --- Text-to-Speech Logic ---
        # text_to_speak is already determined above based on translation outcome
        spoken_language_for_log = target_language.lower() if response_translation_status == "success" and target_language.lower() != 'en' else 'en'
        print(f"INFO: Attempting to speak using TTS: '{text_to_speak}' (Intended Language: {spoken_language_for_log})")
        try:
            if text_to_speak and text_to_speak.strip(): # Only speak if there's actual content
                speak_offline(text_to_speak)
                print("INFO: Speaking initiated via speak_offline.")
            else:
                print("INFO: Skipping TTS because text_to_speak is empty or whitespace.")
        except Exception as speak_error:
            print(f"ERROR: Error during speak_offline call: {speak_error}")
            traceback.print_exc() # Print full traceback for TTS error


        # --- Prepare Final Response ---
        response_content = {
            "original_english_caption": english_caption,
            "translated_caption": response_translated_caption,
            "target_language_requested": target_language,
            "translation_status": response_translation_status,
        }
        if response_translation_error: # Only add error key if there is an error
            response_content["translation_error"] = response_translation_error
        
        print(f"DEBUG: Final response_content to be sent to UI: {response_content}")
        return JSONResponse(content=response_content)

    except Exception as e:
        print(f"FATAL ERROR: Unhandled exception in /caption endpoint: {str(e)}")
        traceback.print_exc()
        # Provide a structured error response
        final_error_response = {
            "error": "Failed to process image or generate caption due to an internal server error.",
            "details": str(e),
            "original_english_caption": english_caption if english_caption else "Caption generation failed or did not complete.",
            "translated_caption": english_caption if english_caption else "Caption generation failed or did not complete.",
            "target_language_requested": target_language, # The language that was requested
            "translation_status": "error_in_processing", # Specific status for this type of error
            "translation_error": "An unexpected server error occurred during the captioning pipeline."
        }
        print(f"DEBUG: Error response_content to be sent to UI: {final_error_response}")
        return JSONResponse(
            status_code=500, # Internal Server Error
            content=final_error_response
        )

if __name__ == "__main__":
    import uvicorn
    print("Starting Vision.ai FastAPI server...")
    # Ensure the host and port match what app.py (API_URL) expects
    # Default API_URL in app.py is "http://127.0.0.1:8000/caption"
    # The server should thus run on 127.0.0.1 (localhost) and port 8000.
    uvicorn.run(app, host="127.0.0.1", port=8000)