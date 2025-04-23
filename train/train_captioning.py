# STEP 1: Load data
from src.captioning.dataset import CaptionDataset
from src.captioning.vocabulary import Vocabulary
from src.captioning.utils import load_captions
from src.captioning.transforms import get_transform
from torch.utils.data import DataLoader
import os
import torch

# Paths
image_folder = "Data/images/Flicker8k_Dataset"
caption_file = "Data/Flickr8k_text/Flickr8k.token.txt"

# Load and clean
caption_dict = load_captions(caption_file)
vocab = Vocabulary(freq_threshold=5)
vocab.build_vocabulary(caption_dict)
transform = get_transform()
dataset = CaptionDataset(image_folder=image_folder, caption_dict=caption_dict, vocab=vocab, transform=transform)
data_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, collate_fn=dataset.collate_fn)

# STEP 2: Model, Loss, Optimizer
from src.captioning.encoder_decoder import CaptioningModel
import torch.nn as nn
import torch.optim as optim

vocab_size = len(vocab)
embed_size = 512
num_heads = 8
hidden_dim = 2048
num_layers = 3
dropout = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CaptioningModel(
    vocab_size=vocab_size,
    embed_size=embed_size,
    num_heads=num_heads,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    dropout=dropout,
    max_len=100
).to(device)
model.train()

# Define loss and optimizer
pad_idx = vocab.stoi["<PAD>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=3e-4)
