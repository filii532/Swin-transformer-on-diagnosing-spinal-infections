
from torch import nn
# from .resnet import ResNet, BasicBlock, Bottleneck
# from .efficientnet import EfficientNet
from .swintransformer import SwinTransformer
# from .ViT import VisionTransformer

# ### ResNet ###
# def resnet34(num_classes=1000, include_top=True):
#     return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

# def resnet50(num_classes=1000, include_top=True):
#     return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)

# def resnet101(num_classes=1000, include_top=True):
#     return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

# def resnet152(num_classes=1000, include_top=True):
#     return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, include_top=include_top)

# ### EfficientNet ###
# def efficientnet_b0(num_classes=1000):
#     return EfficientNet(width_coefficient=1.0,
#                         depth_coefficient=1.0,
#                         dropout_rate=0.2,
#                         num_classes=num_classes)


### SwinTransformer ###
def swin_tiny(num_classes: int = 1000, **kwargs):
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 6, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            attn_type = 'AAAA',
                            **kwargs)
    return model

def swin_small(num_classes: int = 1000, **kwargs):
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=96,
                            depths=(2, 2, 18, 2),
                            num_heads=(3, 6, 12, 24),
                            num_classes=num_classes,
                            attn_type = 'BBBB',
                            **kwargs)
    return model

def swin_base(num_classes: int = 21841, **kwargs):
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=128,
                            depths=(2, 2, 18, 2),
                            num_heads=(4, 8, 16, 32),
                            num_classes=num_classes,
                            attn_type = 'BBBB',
                            **kwargs)
    return model

def swin_large(num_classes: int = 21841, **kwargs):
    model = SwinTransformer(in_chans=3,
                            patch_size=4,
                            window_size=7,
                            embed_dim=192,
                            depths=(2, 2, 18, 2),
                            num_heads=(6, 12, 24, 48),
                            num_classes=num_classes,
                            attn_type = 'AAAA',
                            **kwargs)
    return model

# ### VisionTransformer ###
# def vit_base(num_classes: int = 21843, has_logits: bool = True):
#     model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=768,
#                               depth=12,
#                               num_heads=12,
#                               representation_size=768 if has_logits else None,
#                               num_classes=num_classes)
#     return model

# def vit_large(num_classes: int = 21843, has_logits: bool = True):
#     model = VisionTransformer(img_size=224,
#                               patch_size=16,
#                               embed_dim=1024,
#                               depth=24,
#                               num_heads=16,
#                               representation_size=1024 if has_logits else None,
#                               num_classes=num_classes)
#     return model

# def vit_huge_14(num_classes: int = 21843, has_logits: bool = True):
#     model = VisionTransformer(img_size=224,
#                               patch_size=14,
#                               embed_dim=1280,
#                               depth=32,
#                               num_heads=16,
#                               representation_size=1280 if has_logits else None,
#                               num_classes=num_classes)
#     return model
