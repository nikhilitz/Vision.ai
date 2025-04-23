import os
import sys
import torch
import unittest
from src.captioning.encoder_decoder import CNNEncoder

# Set project root path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


class TestCNNEncoder(unittest.TestCase):
    def setUp(self):
        self.embed_size = 512
        self.encoder = CNNEncoder(embed_size=self.embed_size)
        self.encoder.eval()  # no gradients for testing

        # Create a dummy batch of 2 images, each 3x224x224 (standard input for ResNet)
        self.dummy_images = torch.randn(2, 3, 224, 224)

    def test_encoder_output_shape(self):
        with torch.no_grad():
            output = self.encoder(self.dummy_images)
        
        print("Encoder output shape:", output.shape)
        self.assertEqual(output.shape, (2, self.embed_size))  # [batch_size, embed_size]

if __name__ == '__main__':
    unittest.main()
