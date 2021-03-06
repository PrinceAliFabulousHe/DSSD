from ... import registry
from . import resnet

__all__ = ['resnet101']


@registry.BACKBONES.register('resnet101')
def resnet101(cfg, pretrained=True):
    return resnet.resnet101(pretrained=pretrained)


@registry.BACKBONES.register('resnet50')
def resnet50(cfg, pretrained=True):
    return resnet.resnet50(pretrained=pretrained)
