import torch
from torch import nn

from vit.embeddings import PatchEmbedding, PositionalEmbedding
from vit.transformer import Transformer


class ViT(nn.Module):
    def __init__(self,
                 image_size,
                 patch_size=16,
                 in_channels=3,
                 embed_dim=768,
                 num_layers=12,
                 num_heads=12,
                 hidden_dim=3072,
                 atten_drop=0.,
                 proj_drop=0.1,
                 head = None):
        super(ViT, self).__init__()
        self.embed_dim = embed_dim
        seq_length = (image_size // patch_size) ** 2 + 1
        self.patch_embedding = PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embendding = PositionalEmbedding(seq_length, embed_dim)
        self.transformer = Transformer(num_layers,
                                       embed_dim,
                                       num_heads,
                                       hidden_dim,
                                       atten_drop,
                                       proj_drop)

        self.head = head if head is not None else nn.Identity()


    def forward(self, x):
        b, _, _, _ = x.size()
        x = self.patch_embedding(x)
        x = torch.cat([self.cls_token.expand(b, -1, -1), x], dim=1)
        x = self.pos_embendding(x)
        x = self.transformer(x)

        return self.head(x[:, 0])







