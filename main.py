import torch
import numpy as np
from vit.configs import ViT_B16_Config
from vit.heads import MLPHead
from vit.vit import ViT

if __name__ == "__main__":
    embed_dim = ViT_B16_Config['embed_dim']
    torch.autograd.set_detect_anomaly(True)
    head = MLPHead(embed_dim, 1)
    vit = ViT(image_size=32, head=head, **ViT_B16_Config)

    inpt = torch.rand((1, 3, 32, 32))

    ls = torch.nn.CrossEntropyLoss(reduction='mean')
    res = vit(inpt)
    loss = ls(res, torch.from_numpy(np.array([[0.]])))
    loss.backward()

