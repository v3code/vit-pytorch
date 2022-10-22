from torch import nn


class MLP(nn.Module):
    def __init__(self, embed_dim, mlp_dim,
                 dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.mlp(x)
