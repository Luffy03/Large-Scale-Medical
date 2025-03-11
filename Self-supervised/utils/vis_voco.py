import json

import cv2
import numpy as np
import scipy.ndimage as ndimage
import torch
import os
import SimpleITK as sitk
from tqdm import tqdm
from monai import data, transforms
from monai.transforms import *

from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import random
from model.backbone.dinov2 import *
from model.backbone.dinov2_3D import *
from monai.networks.nets import SwinUNETR
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
from monai.networks.nets.swin_unetr import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final

from monai.networks.blocks import MLPBlock as Mlp
from monai.networks.blocks import PatchEmbed, UnetOutBlock, UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers import DropPath, trunc_normal_
from monai.utils import ensure_tuple_rep, look_up_option, optional_import


def norm(img):
    min, max = img.min(), img.max()
    img = (img - min)/(max - min + 1e-6)
    return img


def color_map(dataset='pascal'):
    cmap = np.zeros((256, 3), dtype='uint8')

    if dataset == 'pascal' or dataset == 'coco':
        def bitget(byteval, idx):
            return (byteval & (1 << idx)) != 0

        for i in range(256):
            r = g = b = 0
            c = i
            for j in range(8):
                r = r | (bitget(c, 0) << 7-j)
                g = g | (bitget(c, 1) << 7-j)
                b = b | (bitget(c, 2) << 7-j)
                c = c >> 3

            cmap[i] = np.array([r, g, b])

    elif dataset == 'cityscapes':
        cmap[0] = np.array([128, 64, 128])
        cmap[1] = np.array([244, 35, 232])
        cmap[2] = np.array([70, 70, 70])
        cmap[3] = np.array([102, 102, 156])
        cmap[4] = np.array([190, 153, 153])
        cmap[5] = np.array([153, 153, 153])
        cmap[6] = np.array([250, 170, 30])
        cmap[7] = np.array([220, 220, 0])
        cmap[8] = np.array([107, 142, 35])
        cmap[9] = np.array([152, 251, 152])
        cmap[10] = np.array([70, 130, 180])
        cmap[11] = np.array([220, 20, 60])
        cmap[12] = np.array([255,  0,  0])
        cmap[13] = np.array([0,  0, 142])
        cmap[14] = np.array([0,  0, 70])
        cmap[15] = np.array([0, 60, 100])
        cmap[16] = np.array([0, 80, 100])
        cmap[17] = np.array([0,  0, 230])
        cmap[18] = np.array([119, 11, 32])

        cmap[19] = np.array([0, 0, 0])
        cmap[255] = np.array([0, 0, 0])

    return cmap


def random_crop(image, label, crop_size=(128, 128, 128)):

    depth, height, width = image.shape

    if (depth < crop_size[0]) or (height < crop_size[1]) or (width < crop_size[2]):
        raise ValueError("Crop size must be smaller than the volume size.")

    z_start = random.randint(0, depth - crop_size[0])
    y_start = random.randint(0, height - crop_size[1])
    x_start = random.randint(0, width - crop_size[2])

    cropped_image = image[z_start:z_start + crop_size[0],
                     y_start:y_start + crop_size[1],
                     x_start:x_start + crop_size[2]]

    cropped_label = label[z_start:z_start + crop_size[0],
                     y_start:y_start + crop_size[1],
                     x_start:x_start + crop_size[2]]
    return cropped_image, cropped_label


def vis():
    import argparse
    parser = argparse.ArgumentParser(description="PyTorch Training")
    roi = 128
    parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--feature_size", default=48, type=int, help="embedding size")
    parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")

    args = parser.parse_args()

    path = './cache_data/BTCV'
    ls = os.listdir(path)
    save_path = './cases_voco'

    import random
    random.shuffle(ls)
    # ls.sort()

    for i in ls:
        # i = ls[1]
        # print(i)

        data = torch.load(os.path.join(path, i))
        img = data['image'][0].cpu().numpy()
        lab = data['label'][0].cpu().numpy()
        img, lab = random_crop(img, lab)
        print('input shape:', img.shape, lab.shape)

        img, lab = np.rot90(img), np.rot90(lab)
        print(np.unique(lab))

        # swin out
        swin = get_swin_output(img, args)
        swin_no = get_swin_output(img, args, False)

        swin_une = get_swin_output(img, args, path='./pretrained/self_supervised_nv_swin_unetr_5050.pt')
        swin_cli = get_swin_output(img, args, path='./pretrained/supervised_clip_driven_universal_swin_unetr_2100.pth')
        swin_sup = get_swin_output(img, args, path='./pretrained/supervised_suprem_swinunetr_2100.pth')

        h, w, c = img.shape
        cmap = color_map()

        img = norm(img)

        for j in range(c):
            im = img[:, :, j]
            la = lab[:, :, j]
            voco = swin[:, :, j]

            sw_no = swin_no[:, :, j]
            sw_une = swin_une[:, :, j]

            sw_cli = swin_cli[:, :, j]
            sw_sup = swin_sup[:, :, j]

            cls_set = list(np.unique(la))

            if len(cls_set) > 1:
                print(list(np.unique(la)))

                im = (255 * im).astype(np.uint8)
                la = Image.fromarray(la.astype(np.uint8), mode='P')
                la.putpalette(cmap)

                fig, axs = plt.subplots(1, 6, figsize=(20, 5))
                axs[0].imshow(im, cmap='gray')
                axs[0].axis("off")

                axs[1].imshow(sw_no)
                axs[1].axis("off")

                axs[2].imshow(sw_une)
                axs[2].axis("off")

                axs[3].imshow(sw_cli)
                axs[3].axis("off")

                axs[4].imshow(sw_sup)
                axs[4].axis("off")

                axs[5].imshow(voco)
                axs[5].axis("off")

                plt.tight_layout()
                plt.show()
                plt.close()

                name = np.random.randint(0, 1000)
                print(name)
                save_case_path = os.path.join(save_path, str(name))
                if not os.path.exists(save_case_path):
                    os.makedirs(save_case_path)

                cv2.imwrite(os.path.join(save_case_path, 'img.png'), im)

                save_by_plot(sw_no, os.path.join(save_case_path, 'scratch.png'))

                save_by_plot(sw_une, os.path.join(save_case_path, 'swinunetr.png'))
                save_by_plot(sw_cli, os.path.join(save_case_path, 'clip.png'))
                save_by_plot(sw_sup, os.path.join(save_case_path, 'suprem.png'))
                save_by_plot(voco, os.path.join(save_case_path, 'voco.png'))
                break


def save_by_plot(im, path):
    plt.imshow(im)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process(im):
    im = torch.from_numpy(im)
    im = im.unsqueeze(0)
    im = torch.concatenate([im, im, im], 0)
    c, h, w = im.size()
    im = im.view(1, 3, h, w)
    # im = F.interpolate(im, size=(280, 280), mode='bilinear')
    im = F.interpolate(im, size=(98, 98), mode='bilinear')
    return im.cuda()


def get_swin_output(input, args, pretrained=True, path='./pretrained/VoComni_B.pt'):
    im = input.copy()
    x, y, z = im.shape
    im = torch.from_numpy(im)
    im = im.unsqueeze(0).unsqueeze(0).cuda()
    im = F.interpolate(im, size=(128,128,128), mode='trilinear')
    model = Swin(args)

    if pretrained:
        model_dict = torch.load(path, map_location=torch.device('cpu'))
        model = load(model, model_dict)

    model.cuda()
    model.eval()
    with torch.no_grad():
        print(im.shape)
        # output = model.swinViT(im)[-3]

        output = model(im)
        output = output.mean(1).unsqueeze(1)
        print(output.shape)
        output = F.interpolate(output, size=(x, y, z), mode='trilinear')

    output = output[0][0].data.cpu().numpy()
    output = (norm(output) * 255).astype(np.uint8)
    return output


def load(model, model_dict):
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    elif "network_weights" in model_dict.keys():
        state_dict = model_dict["network_weights"]
    elif "net" in model_dict.keys():
        state_dict = model_dict["net"]
    elif "student" in model_dict.keys():
        state_dict = model_dict["student"]
    else:
        state_dict = model_dict

    if "module." in list(state_dict.keys())[0]:
        print("Tag 'module.' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("module.", "")] = state_dict.pop(key)

    if "backbone." in list(state_dict.keys())[0]:
        print("Tag 'backbone.' found in state dict - fixing!")
    for key in list(state_dict.keys()):
        state_dict[key.replace("backbone.", "")] = state_dict.pop(key)

    if "swin_vit" in list(state_dict.keys())[0]:
        print("Tag 'swin_vit' found in state dict - fixing!")
        for key in list(state_dict.keys()):
            state_dict[key.replace("swin_vit", "swinViT")] = state_dict.pop(key)

    current_model_dict = model.state_dict()

    # for k in current_model_dict.keys():
    #     if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
    #         print('load:', k)

    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}

    model.load_state_dict(new_state_dict, strict=True)

    return model


class Swin(nn.Module):
    def __init__(self, args):
        super(Swin, self).__init__()
        patch_size = ensure_tuple_rep(2, args.spatial_dims)
        # print(patch_size)
        window_size = ensure_tuple_rep(7, args.spatial_dims)
        # print(window_size)
        self.swinViT = SwinTransformer(
            in_chans=args.in_channels,
            embed_dim=args.feature_size,
            window_size=window_size,
            patch_size=patch_size,
            depths=[2, 2, 2, 2],
            num_heads=[3, 6, 12, 24],
            mlp_ratio=4.0,
            qkv_bias=True,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=args.dropout_path_rate,
            norm_layer=torch.nn.LayerNorm,
            use_checkpoint=args.use_checkpoint,
            spatial_dims=args.spatial_dims,
            use_v2=True,
        )
        norm_name = 'instance'
        self.encoder1 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.in_channels,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder2 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder3 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=2 * args.feature_size,
            out_channels=2 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder4 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=4 * args.feature_size,
            out_channels=4 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.encoder10 = UnetrBasicBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=16 * args.feature_size,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=16 * args.feature_size,
            out_channels=8 * args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder4 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size * 8,
            out_channels=args.feature_size * 4,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder3 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size * 4,
            out_channels=args.feature_size * 2,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size * 2,
            out_channels=args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

        self.decoder1 = UnetrUpBlock(
            spatial_dims=args.spatial_dims,
            in_channels=args.feature_size,
            out_channels=args.feature_size,
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=True,
        )

    def forward(self, x_in):
        hidden_states_out = self.swinViT(x_in)
        # for i in hidden_states_out:
        #     print('hidden_states_out:', i.shape)

        enc0 = self.encoder1(x_in)
        enc1 = self.encoder2(hidden_states_out[0])
        enc2 = self.encoder3(hidden_states_out[1])
        enc3 = self.encoder4(hidden_states_out[2])
        dec4 = self.encoder10(hidden_states_out[4])

        encs = [enc0, enc1, enc2, enc3, dec4]

        dec3 = self.decoder5(dec4, hidden_states_out[3])
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        out = self.decoder1(dec0, enc0)
        # print('out:', out.shape)

        return out


if __name__ == '__main__':
    vis()

