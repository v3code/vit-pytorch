from vit.vit import ViT


class VitFactory:

    @staticmethod
    def create_from_config(config):
        return ViT(*config)

