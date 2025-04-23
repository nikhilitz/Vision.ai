import torch
import torch.nn as nn
import torchvision.models as models
import math

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super(CNNEncoder, self).__init__()

        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=True)

        # Remove the final fully connected (classification) layer
        modules = list(resnet.children())[:-1]  # drop the last FC layer
        self.resnet = nn.Sequential(*modules)

        # Freeze all ResNet layers (so we only train the projection layer)
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Add a linear layer to convert ResNet's 2048-dim output to embed_size
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        # Extract CNN features
        features = self.resnet(images)               # shape: [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # flatten → [B, 2048]

        # Project to desired embedding size
        features = self.fc(features)                 # → [B, embed_size]
        features = self.bn(features)
        return features

class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1, max_len=100):
        super(TransformerDecoder, self).__init__()

        self.embed_size = embed_size

        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        self.position_encoding = PositionalEncoding(embed_size, max_len=max_len)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers
        )

        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
