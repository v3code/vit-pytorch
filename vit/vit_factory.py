from vit.vit import ViT


class VitFactory:

    @staticmethod
    def create_from_config(config, img_size, head):
        return ViT(image_size=img_size, head=head, *config)

