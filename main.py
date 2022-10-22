import torch

from vit.configs import ViT_B16_Config
from vit.heads import MLPHead
from vit.vit import ViT

if __name__ == "__main__":
    embed_dim = ViT_B16_Config['embed_dim']
    head = MLPHead(embed_dim, 10)
    vit = ViT(image_size=32, head=head, **ViT_B16_Config)

    inpt = torch.rand((1, 3, 32, 32))
    print(vit(inpt))

