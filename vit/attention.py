from torch import nn


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, atten_dropout=0., proj_dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = 1. / head_dim ** 0.5

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(atten_dropout)

        self.proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(proj_dropout)
        )

    def split_qkv(self, qkv_element, b, n, c):
        qkv_element = (qkv_element
                       .reshape(b,
                                n,
                                3,
                                self.num_heads,
                                c // self.num_heads)
                       .premute(2, 0, 3, 1, 4))
        return qkv_element.unbind(0)

    def forward(self, x):
        qkv = self.qkv(x)
        queue, key, value = self.split_qkv(qkv, *x.shape)

        attention = (queue @ key.transpose(-1, -2)) * self.scale
        attention = self.attn_drop(attention.softmax(dim=-1))

        x = (attention @ value).transpose(1, 2).reshape(x.shape)

        return self.proj(x)

