import os
import sys
import unittest
from PIL import Image
import torch
from torchvision import transforms

# Append project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.captioning.vocabulary import Vocabulary
from src.captioning.dataset import CaptionDataset

class TestCaptionDataset(unittest.TestCase):

    def setUp(self):
        # Create dummy caption dict
        self.caption_dict = {
            "sample.jpg": ["a dog is playing"]
        }

        # Build vocabulary manually
        self.vocab = Vocabulary(freq_threshold=1)
        self.vocab.build_vocabulary(["a dog is playing"])

        # Dummy image path
        self.image_folder = "Data/images/"

        # Ensure a dummy image exists
        dummy_image_path = os.path.join(self.image_folder, "sample.jpg")
        if not os.path.exists(dummy_image_path):
            img = Image.new("RGB", (224, 224), color='white')
            os.makedirs(self.image_folder, exist_ok=True)
            img.save(dummy_image_path)

        # Dummy transform (same as model expects)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        # Create dataset
        self.dataset = CaptionDataset(
            image_folder=self.image_folder,
            caption_dict=self.caption_dict,
            vocabulary=self.vocab,
            transform=self.transform
        )

    def test_dataset_length(self):
        self.assertEqual(len(self.dataset), 1)

    def test_dataset_output_format(self):
        image, caption_tensor = self.dataset[0]
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(caption_tensor, torch.Tensor)
        self.assertEqual(image.shape, torch.Size([3, 224, 224]))
        self.assertGreaterEqual(len(caption_tensor), 3)  # At least <sos> token + words + <eos>

    def test_caption_numericalization(self):
        _, caption_tensor = self.dataset[0]
        decoded_words = [self.vocab.itos.get(idx.item(), "<unk>") for idx in caption_tensor]
        # Should start with <sos> and end with <eos>
        self.assertEqual(decoded_words[0], "<sos>")
        self.assertEqual(decoded_words[-1], "<eos>")

if __name__ == '__main__':
    unittest.main()
