# Path: Vision.ai/src/captioning/encoder_decoder.py

import torch
import torch.nn as nn
import math
from torchvision.models import resnet50, ResNet50_Weights

# =========================
# Positional Encoding
# =========================

class PositionalEncoding(nn.Module):
    """
    Injects sinusoidal positional encoding into token embeddings.
    This provides the model with information about the position of tokens in a sequence.
    """
    def __init__(self, embed_size, max_len=5000):
        """
        :param embed_size: Dimension of the input embeddings (d_model).
        :param max_len: Maximum possible length of the input sequence.
        """
        super(PositionalEncoding, self).__init__()
        # Create the positional encoding matrix
        pe = torch.zeros(max_len, embed_size)
        # Create a tensor of positions [0, 1, ..., max_len-1]
        position = torch.arange(0, max_len).unsqueeze(1) # Shape: [max_len, 1]
        # Calculate the division term for the sinusoidal functions
        # 1 / (10000^(2i / d_model))
        div_term = torch.exp(torch.arange(0, embed_size, 2) * (-math.log(10000.0) / embed_size)) # Shape: [embed_size / 2]

        # Apply sine to even indices and cosine to odd indices
        # pe[position, 2i] = sin(position / div_term[i])
        # pe[position, 2i+1] = cos(position / div_term[i])
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension at the beginning for easier addition to input embeddings
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, embed_size]
        # Register pe as a buffer. It's part of the module state but not a learnable parameter.
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Adds positional encoding to the input tensor.
        :param x: Input tensor (e.g., token embeddings) of shape [B, T, E]
        :return: Tensor with positional encoding added, shape [B, T, E]
        """
        # Add the positional encoding to the input.
        # Slice the stored pe tensor to match the current sequence length x.size(1)
        # Use .detach() as gradients should not flow back to the positional encoding buffer
        x = x + self.pe[:, :x.size(1)].detach()
        return x


# =========================
# CNN Encoder using ResNet50
# =========================

class CNNEncoder(nn.Module):
    """
    CNN Encoder using a pre-trained ResNet model to extract image features.
    Allows selective fine-tuning of specified ResNet layers.
    A linear layer maps the features to the desired embedding size.
    """
    # Add fine_tune_layers parameter to constructor
    def __init__(self, embed_size, fine_tune_layers=None):
        """
        :param embed_size: The dimension of the output image features (d_model for the Transformer).
        :param fine_tune_layers: List of strings specifying which ResNet layers to unfreeze (e.g., ['layer4', 'layer3']).
                                 If None or empty list, the entire ResNet remains frozen.
        """
        super(CNNEncoder, self).__init__()
        # Load a pre-trained ResNet50 model with default ImageNet weights
        weights = ResNet50_Weights.DEFAULT
        resnet = resnet50(weights=weights)

        # Get individual layer access to allow selective freezing/unfreezing
        # Store them as attributes of this module
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1 # Bottleneck layers
        self.layer2 = resnet.layer2 # Bottleneck layers
        self.layer3 = resnet.layer3 # Bottleneck layers
        self.layer4 = resnet.layer4 # Bottleneck layers
        self.avgpool = resnet.avgpool # Keep the average pooling layer

        # --- Freezing/Unfreezing Logic ---
        # Freeze all ResNet parameters initially
        # We iterate through *all* named parameters of the ResNet components we've added
        for name, param in self.named_parameters():
             # Ensure we only freeze parameters from the ResNet components, not the new fc/bn layers
             # Check if the parameter name starts with a ResNet component name
             if any(name.startswith(resnet_part) for resnet_part in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4', 'avgpool']):
                  param.requires_grad = False
                  # print(f"Froze: {name}") # Optional: for debugging


        # Unfreeze specified layers for fine-tuning
        if fine_tune_layers is not None:
             for layer_name in fine_tune_layers:
                  # Get the layer module by name (e.g., self.layer4)
                  layer = getattr(self, layer_name, None) # Use getattr with default None to avoid error if layer_name is invalid
                  if layer is not None:
                       # Unfreeze all parameters within this layer
                       for param in layer.parameters():
                            param.requires_grad = True
                       print(f"✅ Unfroze layer for fine-tuning: {layer_name}")
                  else:
                       print(f"Warning: Layer '{layer_name}' not found in ResNet components. Cannot fine-tune.")


        # Add a linear layer to map ResNet output features (2048 for ResNet50 after AvgPool) to embed_size
        # We get the input feature size from the original resnet.fc layer
        self.fc = nn.Linear(resnet.fc.in_features, embed_size) # resnet.fc.in_features is 2048 for ResNet50

        # Add Batch Normalization after the linear layer
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)


    def forward(self, images):
        """
        Forward pass to extract image features.
        :param images: Input image tensors, shape [B, 3, H, W] (e.g., 224, 224)
        :return: Image feature vectors, shape [B, embed_size]
        """
        # Pass images through the ResNet layers sequentially
        # No need for torch.no_grad() around the whole pass if some layers are unfrozen
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) # Output shape depends on input and layer structure
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Pass through Adaptive Average Pooling
        features = self.avgpool(x)  # Output after AvgPool: [B, 2048, 1, 1]

        # Flatten the features from [B, 2048, 1, 1] to [B, 2048]
        features = features.view(features.size(0), -1)

        # Pass through the linear layer
        features = self.fc(features)  # Output: [B, embed_size]

        # Pass through BatchNorm
        features = self.bn(features) # Output: [B, embed_size]
        return features


# =========================
# Transformer Decoder
# =========================

class TransformerDecoder(nn.Module):
    """
    Transformer Decoder to generate caption sequences based on image features.
    """
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1, max_len=100):
        """
        :param vocab_size: The size of the output vocabulary.
        :param embed_size: Dimension of token embeddings and model features (d_model).
        :param num_heads: Number of attention heads in the Transformer layers.
        :param hidden_dim: Dimension of the feedforward network in the Transformer layers.
        :param num_layers: Number of Transformer decoder layers.
        :param dropout: Dropout rate.
        :param max_len: Maximum possible length of the target sequence for positional encoding.
        """
        super(TransformerDecoder, self).__init__()

        self.embed_size = embed_size
        # Word embedding layer: maps token indices to dense vectors
        self.word_embedding = nn.Embedding(vocab_size, embed_size)
        # Positional encoding layer: adds positional information to embeddings
        self.position_encoding = PositionalEncoding(embed_size, max_len=max_len)

        # Define a single Transformer Decoder Layer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_size,         # Input feature dimension (must be divisible by num_heads)
            nhead=num_heads,            # Number of attention heads
            dim_feedforward=hidden_dim, # Dimension of the feedforward network
            dropout=dropout,            # Dropout rate
            batch_first=False           # Standard Transformer API is seq_len first
        )
        # Stack multiple Decoder Layers to form the Transformer Decoder
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_layers       # Number of decoder layers
        )

        # Final linear layer: maps decoder output features back to vocabulary size (logits)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        # Dropout layer (applied to input embeddings)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt_seq, memory, tgt_mask):
        """
        Forward pass for the Transformer Decoder.
        :param tgt_seq: [B, T] token indices of the target sequence (captions).
                        During training, this is the shifted caption sequence (teacher forcing).
                        During inference, this is the sequence generated so far.
        :param memory: [B, embed_size] image embeddings (output from encoder). This acts as the memory for cross-attention.
        :param tgt_mask: [T, T] mask to prevent looking ahead (causal mask). Generated in train/inference script.
        :return: [B, T, vocab_size] logits over the vocabulary for each step in the sequence.
        """
        # Get word embeddings: [B, T] -> [B, T, E]
        tgt_emb = self.word_embedding(tgt_seq)
        # Add positional encoding: [B, T, E] -> [B, T, E]
        tgt_emb = self.position_encoding(tgt_emb)
        # Apply dropout: [B, T, E] -> [B, T, E]
        tgt_emb = self.dropout(tgt_emb)

        # Transpose target embeddings for Transformer input shape: [B, T, E] -> [T, B, E]
        # Transformer expects (sequence_length, batch_size, feature_dimension)
        tgt_emb = tgt_emb.transpose(0, 1)

        # Prepare memory for Transformer Decoder: [B, E] -> [1, B, E]
        # The memory from the encoder (image features) is a single vector per batch item.
        # Transformer's memory input expects shape (memory_sequence_length, batch_size, feature_dimension).
        # Here, memory_sequence_length is 1.
        memory = memory.unsqueeze(0) # Shape: [B, E] -> [1, B, E]


        # Pass through the Transformer Decoder layers
        # tgt: [T, B, E], memory: [1, B, E], tgt_mask: [T, T]
        # memory_mask and memory_key_padding_mask are not used here as memory sequence length is 1
        output = self.transformer_decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_mask=tgt_mask # Causal mask for target sequence (prevents attending to future tokens)
        )  # Output shape: [T, B, E]

        # Pass through the final linear layer to get logits over the vocabulary
        output = self.fc_out(output)   # Output shape: [T, B, vocab_size]

        # Transpose output back to [B, T, vocab_size] for compatibility with loss function and standard usage
        return output.transpose(0, 1)


# =========================
# Combined Captioning Model
# =========================

class CaptioningModel(nn.Module):
    """
    Combines the CNN Encoder and Transformer Decoder for the image captioning task.
    Allows specifying which encoder layers to fine-tune.
    """
    # Add fine_tune_encoder_layers parameter
    def __init__(self, vocab_size, embed_size, num_heads, hidden_dim, num_layers, dropout=0.1, max_len=100, fine_tune_encoder_layers=None):
        """
        :param vocab_size: The size of the output vocabulary.
        :param embed_size: Dimension of embeddings and model features (d_model).
        :param num_heads: Number of attention heads.
        :param hidden_dim: Dimension of the feedforward network.
        :param num_layers: Number of Transformer layers.
        :param dropout: Dropout rate.
        :param max_len: Maximum sequence length for positional encoding.
        :param fine_tune_encoder_layers: List of strings for ResNet layers to unfreeze (e.g., ['layer4']).
        """
        super(CaptioningModel, self).__init__()
        # Instantiate the Encoder (Image Feature Extractor), passing the fine_tune_layers argument
        self.encoder = CNNEncoder(embed_size, fine_tune_layers=fine_tune_encoder_layers)
        # Instantiate the Decoder (Caption Sequence Generator)
        self.decoder = TransformerDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            max_len=max_len # Pass max_len to the decoder for positional encoding setup
        )

    def forward(self, images, captions, tgt_mask):
        """
        Forward pass for the entire Captioning Model.
        :param images: [B, 3, H, W] input image tensors.
        :param captions: [B, T] token indices of the target captions (shifted input for teacher forcing).
        :param tgt_mask: [T, T] causal mask for the target sequence.
        :return: [B, T, vocab_size] logits over the vocabulary for each step in the sequence.
        """
        # 1. Encode the images to get compressed feature representations
        # The encoder's requires_grad settings determine which parts are trained
        image_embeddings = self.encoder(images) # Output shape: [B, embed_size]

        # 2. Decode the captions using the image embeddings as context (memory)
        # The decoder predicts the next token based on the image and the tokens seen so far.
        # captions are the input sequence (shifted) and tgt_mask enforces causality.
        logits = self.decoder(captions, image_embeddings, tgt_mask) # Output shape: [B, T, vocab_size]

        return logits