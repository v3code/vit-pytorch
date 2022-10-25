from torch import nn

from vit.attention import Attention
from vit.mlp import MLP


class Block(nn.Module):
    def __init__(self,
                 embed_dim,
                 num_heads,
                 mlp_dim,
                 atten_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.atten = Attention(embed_dim, num_heads, atten_dropout=atten_drop, proj_dropout=proj_drop)
        self.mlp = MLP(embed_dim, mlp_dim, dropout=proj_drop)

    def forward(self, x):
        x = x + self.atten(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self,
                 num_layers,
                 embed_dim,
                 mlp_dim,
                 num_heads=8,
                 atten_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Dropout(proj_drop),
            nn.LayerNorm(embed_dim)
        )
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_dim, atten_drop, proj_drop)
            for _ in range(num_layers)])

    def forward(self, x):
        x = self.preprocess(x)
        for block in self.blocks:
            x = block(x)
        return x

