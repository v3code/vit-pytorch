from torch import nn


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.Dropout(p=dropout),
            nn.GELU(),
            nn.Linear(hidden_features, out_features),
            nn.Dropout(p=dropout)
        )

    def forward(self, x):
        return self.mlp(x)
