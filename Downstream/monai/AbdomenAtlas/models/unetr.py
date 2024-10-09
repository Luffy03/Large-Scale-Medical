"""Jia-Xin ZHUANG.
"""
from typing import Tuple, Union
import torch
import torch.nn as nn
from monai.networks.blocks import UnetrBasicBlock, UnetrPrUpBlock, UnetrUpBlock
from monai.networks.blocks.dynunet_block import UnetOutBlock

import models_3dvit
import sys


class UNETR(nn.Module):
    """
    UNETR based on: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        img_size: Tuple[int, int, int],
        feature_size: int = 16,
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        num_layer: int=12,
        pos_embed: str = "perceptron",
        norm_name: Union[Tuple, str] = "instance",
        conv_block: bool = False,
        res_block: bool = True,
        dropout_rate: float = 0.0,
        model_name=None
    ) -> None:
        """
        Args:
            in_channels: dimension of input channels.
            out_channels: dimension of output channels.
            img_size: dimension of input image.
            feature_size: dimension of network feature size.
            hidden_size: dimension of hidden layer.
            mlp_dim: dimension of feedforward layer.
            num_heads: number of attention heads.
            pos_embed: position embedding layer type.
            norm_name: feature normalization type and arguments.
            conv_block: bool argument to determine if convolutional block is used.
            res_block: bool argument to determine if residual block is used.
            dropout_rate: faction of the input units to drop.

        Examples::

            # for single channel input 4-channel output with patch size of (96,96,96), feature size of 32 and batch norm
            >>> net = UNETR(in_channels=1, out_channels=4, img_size=(96,96,96), feature_size=32, norm_name='batch')

            # for 4-channel input 3-channel output with patch size of (128,128,128), conv position embedding and instance norm
            >>> net = UNETR(in_channels=4, out_channels=3, img_size=(128,128,128), pos_embed='conv', norm_name='instance')

        """

        super().__init__()

        if not (0 <= dropout_rate <= 1):
            raise AssertionError("dropout_rate should be between 0 and 1.")

        if hidden_size % num_heads != 0:
            raise AssertionError("hidden size should be divisible by num_heads.")

        if pos_embed not in ["conv", "perceptron"]:
            raise KeyError(f"Position embedding layer of type {pos_embed} is not supported.")

        self.num_layers = num_layer
        self.patch_size = (16, 16, 16)
        self.feat_size = (
            img_size[0] // self.patch_size[0],
            img_size[1] // self.patch_size[1],
            img_size[2] // self.patch_size[2],
        )
        self.hidden_size = hidden_size
        self.classification = False
        arch = model_name + '_patch16'
        self.vit = models_3dvit.__dict__[arch](
            img_size=img_size,
            in_channels=in_channels,
            global_pool=False,
        )
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 2,
            num_layer=2,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder3 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 4,
            num_layer=1,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.encoder4 = UnetrPrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            num_layer=0,
            kernel_size=3,
            stride=1,
            upsample_kernel_size=2,
            norm_name=norm_name,
            conv_block=conv_block,
            res_block=res_block,
        )
        self.decoder5 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=hidden_size,
            out_channels=feature_size * 8,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 8,
            out_channels=feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 4,
            out_channels=feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=3,
            in_channels=feature_size * 2,
            out_channels=feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.out = UnetOutBlock(spatial_dims=3, in_channels=feature_size, out_channels=out_channels)  # type: ignore

    def proj_feat(self, x, hidden_size, feat_size):
        x = x.view(x.size(0), feat_size[0], feat_size[1], feat_size[2], hidden_size)
        x = x.permute(0, 4, 1, 2, 3).contiguous()
        return x

    def load_from(self, weights):
        with torch.no_grad():
            res_weight = weights
            # copy weights from patch embedding
            for i in weights["state_dict"]:
                print(i)
            self.vit.patch_embedding.position_embeddings.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.position_embeddings_3d"]
            )
            self.vit.patch_embedding.cls_token.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.cls_token"]
            )
            self.vit.patch_embedding.patch_embeddings[1].weight.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.weight"]
            )
            self.vit.patch_embedding.patch_embeddings[1].bias.copy_(
                weights["state_dict"]["module.transformer.patch_embedding.patch_embeddings.1.bias"]
            )

            # copy weights from  encoding blocks (default: num of blocks: 12)
            for bname, block in self.vit.blocks.named_children():
                print(block)
                block.loadFrom(weights, n_block=bname)
            # last norm layer of transformer
            self.vit.norm.weight.copy_(weights["state_dict"]["module.transformer.norm.weight"])
            self.vit.norm.bias.copy_(weights["state_dict"]["module.transformer.norm.bias"])

    def forward(self, x_in):
        x, hidden_states_out = self.vit(x_in)
        enc1 = self.encoder1(x_in)
        x2 = hidden_states_out[0]
        enc2 = self.encoder2(self.proj_feat(x2, self.hidden_size, self.feat_size))
        x3 = hidden_states_out[1]
        enc3 = self.encoder3(self.proj_feat(x3, self.hidden_size, self.feat_size))
        x4 = hidden_states_out[2]
        enc4 = self.encoder4(self.proj_feat(x4, self.hidden_size, self.feat_size))
        dec4 = self.proj_feat(x, self.hidden_size, self.feat_size)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        out = self.decoder2(dec1, enc1)
        logits = self.out(out)
        return logits


def get_model(model_name='vit_base'):
    """Get model.
    """
    if model_name == 'vit_base':
        num_heads = 12
        num_layer= 12
        hidden_size = 768
        feature_size = 48
    elif model_name == 'vit_small':
        num_heads = 6
        num_layer = 12
        hidden_size = 384
        feature_size = 24
    elif model_name == 'vit_tiny':
        num_heads = 3
        num_layer = 12
        hidden_size = 192
        feature_size = 12
    elif model_name == 'vit_large':
        num_heads = 16
        num_layer = 24
        hidden_size = 1152
        feature_size = 96
    elif model_name == 'vit_huge':
        num_heads = 16
        num_layer = 32
        hidden_size = 1344
        feature_size = 192
    else:
        print('Require valid model name')
        sys.exit(-1)

    # if args.feature_size is not None and args.rank == 0:
    #     print('Force feature size to: ', args.feature_size)
    #     feature_size = args.feature_size

    mlp_dim = hidden_size * 4

    in_channels = 1
    out_channels = 10
    roi = 96

    model = UNETR(
        in_channels=in_channels,
        out_channels=out_channels,
        img_size=(roi, roi, roi),
        feature_size=feature_size,
        hidden_size=hidden_size,
        mlp_dim=mlp_dim,
        num_heads=num_heads,
        num_layer=num_layer,
        pos_embed='perceptron',
        norm_name='instance',
        conv_block=True,
        res_block=True,
        dropout_rate=0.0,
        model_name=model_name
    )
    return model


def delete_patch_embed(state_dict):
    for key in list(state_dict.keys()):
        state_dict[key.replace("swinViT.patch_embed", "bad")] = state_dict.pop(key)
    for key in list(state_dict.keys()):
        state_dict[key.replace("encoder1.layer", "bad")] = state_dict.pop(key)

    return state_dict


if __name__=='__main__':
    x = torch.randn(1, 4, 64, 64, 64)

    from monai.networks.nets import SwinUNETR
    model_H = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=4,
        out_channels=10,
        feature_size=192,
        #num_heads=(6, 12, 24, 48),
        depths=(2, 2, 8, 2),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0,
        use_checkpoint=True,
        use_v2=True
    )
    pytorch_total_params = sum(p.numel() for p in model_H.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    model_L = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=4,
        out_channels=10,
        feature_size=96,
        #num_heads=(4, 8, 16, 32),
        depths=(2, 2, 2, 2),
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0,
        use_checkpoint=True,
        use_v2=True
    )
    pytorch_total_params = sum(p.numel() for p in model_L.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)

    model_dict = torch.load("./VoCo_10k.pt", map_location=torch.device('cpu'))
    state_dict = model_dict
    state_dict = delete_patch_embed(state_dict)
    #model.load_state_dict(state_dict, strict=False)

    #y = model(x)
    # print(y.shape)