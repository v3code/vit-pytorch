import torch
from torch import nn


class PatchEmbedding(nn.Module):
    def __init__(self,
                 image_size=224,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768):
        super(PatchEmbedding, self).__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, patch_size, patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class PositionalEmbedding(nn.Module):
    def __init__(self, seq_len, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.embedding = nn.Parameter(torch.zeros(1, seq_len, embed_dim))

    def forward(self, x):
        return x + self.embedding
