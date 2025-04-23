import torch
import torch.nn as nn
import torchvision.models as models
import math

# =========================
# Positional Encoding
# =========================

class PositionalEncoding(nn.Module):
    def __init__(self, embed_size, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, embed_size)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, embed_size]
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x


class CNNEncoder(nn.Module):
    def __init__(self, embed_size):
        super(CNNEncoder, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-1]  # remove final fc layer
        self.resnet = nn.Sequential(*modules)

        for param in self.resnet.parameters():
            param.requires_grad = False  # freeze ResNet

        self.fc = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)  # [B, 2048, 1, 1]
        features = features.view(features.size(0), -1)  # [B, 2048]
        features = self.fc(features)  # [B, embed_size]
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

    def forward(self, tgt_seq, memory, tgt_mask):
        """
        tgt_seq: [B, T] token indices
        memory: [B, embed_size] image embeddings
        tgt_mask: [T, T] mask to prevent looking ahead
        """
        tgt_emb = self.word_embedding(tgt_seq)         # [B, T, E]
        tgt_emb = self.position_encoding(tgt_emb)      # [B, T, E]
        tgt_emb = self.dropout(tgt_emb).transpose(0, 1)  # [T, B, E]

        memory = memory.unsqueeze(1).transpose(0, 1)    # [1, B, E]

        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask
        )  # [T, B, E]

        output = self.fc_out(output)   # [T, B, vocab_size]
        return output.transpose(0, 1)  # [B, T, vocab_size]

class CaptioningModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1, max_len=100):
        super(CaptioningModel, self).__init__()
        self.encoder = CNNEncoder(embed_size)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len
        )

    def forward(self, images, captions, tgt_mask):
        """
        images: [B, 3, 224, 224]
        captions: [B, T] (token indices)
        tgt_mask: [T, T]
        """
        image_embeddings = self.encoder(images)                  # [B, embed_size]
        logits = self.decoder(captions, image_embeddings, tgt_mask)  # [B, T, vocab_size]
        return logits
