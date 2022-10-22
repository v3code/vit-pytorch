from torch import nn

from vit.attention import Attention
from vit.mlp import MLP


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 hidden_dim,
                 atten_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.atten = Attention(dim, num_heads, atten_dropout=atten_drop, proj_dropout=proj_drop)
        self.mlp = MLP(dim, hidden_dim, dropout=proj_drop)

    def forward(self, x):
        x = self.norm1(x)
        x += self.atten(x)
        return x + self.mlp(self.norm2(x))


class Transformer(nn.Module):
    def __init__(self,
                 num_layers,
                 dim,
                 hidden_dim,
                 num_heads=8,
                 atten_drop=0.,
                 proj_drop=0.):
        super().__init__()
        self.preprocess = nn.Sequential(
            nn.Dropout(proj_drop),
            nn.LayerNorm(dim)
        )
        self.blocks = nn.ModuleList([
            Block(dim, num_heads, hidden_dim, atten_drop, proj_drop)
            for _ in range(num_layers)])

    def forward(self, x):
        x = self.preprocess(x)
        for block in self.blocks:
            x = block(x)
        return x

