from vit.vit import ViT


class VitFactory:

    @staticmethod
    def create_from_config(config):
        return ViT(*config)

    @staticmethod
    def create_by_name(name):
        if name == 'vit-l16':
            pass
        pass
