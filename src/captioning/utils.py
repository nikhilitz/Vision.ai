import re
from nltk.translate.bleu_score import sentence_bleu
def clean_caption(caption):
    """
    Cleans caption text by:
    1. Lowercasing
    2. Removing punctuation
    3. Removing extra spaces
    """
    caption=caption.lower()
    caption=re.sub(r"[^a-zA-Z0-9\s]", "", caption)
    caption=re.sub(r"\s+", " ", caption)
    return caption.strip()

def load_captions(file_path):
    """
    Loads captions from a text file and returns a dictionary:
    image_name -> list of cleaned captions
    """
    caption_dict={}
    with open(file_path,'r') as f:
        lines=f.readlines()
    for line in lines:
        line=line.strip()
        if len(line)==0:
            continue
        image_name,caption=line.split('\t')
        image_id=image_name.split('#')[0]
        cleaned=clean_caption(caption)
        
        if image_id not in caption_dict:
            caption_dict[image_id]=[]
        
        caption_dict[image_id].append(cleaned)
    return caption_dict

def get_bleu_score(reference_captions,generated_captions):
    """
    Calculates BLEU score of generated caption compared to reference captions.
    """
    return sentence_bleu([ref.split() for ref in reference_captions],generated_captions)




