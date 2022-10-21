from torch import nn

class MLPHead(nn.Module):
    def __init__(self, embed_dem, num_classes, hidden_size = 796, dropout = 0.):
        super(MLPHead, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dem, hidden_size),
            nn.Dropout(p=dropout),
            nn.LayerNorm(),
            nn.Linear(hidden_size, num_classes),
            nn.Softmax(dim=num_classes)
        )

    def forward(self, x):
        return self.mlp(x)