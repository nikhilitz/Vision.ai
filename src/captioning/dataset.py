import os 
import torch
from PIL import Image
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    def __init__(self,image_folder,caption_dict,vocabulary,transform=None):
        """
        :param image_folder: Path to image directory
        :param caption_dict: Dictionary {image_id: [list of cleaned captions]}
        :param vocabulary: Vocabulary object
        :param transform: torchvision transforms to apply to image
        """
        self.image_folder=image_folder
        self.caption_dict=caption_dict
        self.vocab=vocabulary
        self.transform=transform
        
        # Flatten image-caption pairs into a list for easy indexing
        self.image_caption_pairs=[]
        for image_id,captions in self.caption_dict.items():
            for caption in captions:
                self.image_caption_pairs.append((image_id,caption))

    def __len__(self):
        return len(self.image_caption_pairs)

    def __getitem__(self,index):
        image_id,caption=self.image_caption_pairs[index]
        image_path=os.path.join(self.image_folder,image_id)
        image=Image.open(image_path).convert("RGB")

        if self.transform:
            image=self.transform(image)
        
        #numericalize caption
        numericalized_caption=[self.vocab.stoi['<sos>']]
        numericalized_caption+=self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi['<eos>'])

        return image,torch.tensor(numericalized_caption)

    
