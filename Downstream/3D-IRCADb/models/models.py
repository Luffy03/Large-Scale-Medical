import os

import torch
from monai.networks.nets import SwinUNETR
from models.unet import UNet3D

from models.MiT import MiT
from models.dodnet import DoDNet_UNet3D


def VoCo(args):
    # CVPR 2024 extention
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=True
    )
    if args.feature_size == 48:
        pretrained_path = 'VoComni_B.pt'
        if args.use_ssl_pretrained:
            pretrained_path = 'VoCo_B_SSL_head.pt'

    elif args.feature_size == 96:
        pretrained_path = 'VoComni_L.pt'
        if args.use_ssl_pretrained:
            pretrained_path = 'VoCo_L_SSL_head.pt'

    elif args.feature_size == 192:
        pretrained_path = 'VoComni_H.pt'
        if args.use_ssl_pretrained:
            pretrained_path = 'VoCo_H_SSL_head.pt'

    else:
        print('Error, set args.feature_size in 48, 96, 192')

    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using VoCo pretrained backbone weights !!!!!!!")
    return model


def SuPrem(args, pretrained_path='supervised_suprem_swinunetr_2100.pth'):
    # ICLR 2024
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=False
    )
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using Suprem-ICLR24 pretrained backbone weights !!!!!!!")
    return model


def Swin(args, pretrained_path='self_supervised_nv_swin_unetr_5050.pt'):
    # CVPR 2023
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=False
    )
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using Swin-CVPR23 pretrained backbone weights !!!!!!!")
    return model


def Universal(args, pretrained_path='supervised_clip_driven_universal_swin_unetr_2100.pth'):
    # ICCV 2023
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=args.dropout_path_rate,
        use_checkpoint=args.use_checkpoint,
        use_v2=False
    )
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using clipdriven-ICCV23 pretrained backbone weights !!!!!!!")
    return model


def MG(args, pretrained_path='self_supervised_models_genesis_unet_620.pt'):
    # MedIA 2021
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = UNet3D(n_class=args.out_channels)
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using MG-MedIA21 pretrained backbone weights !!!!!!!")
    return model


def UniMiss(args, pretrained_path='self_supervised_unimiss_nnunet_small_5022.pth'):
    # ECCV 2022 TPAMI 2024
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = MiT(num_classes=args.out_channels)
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using UniMiss-ECCV22-TPAMI24 pretrained backbone weights !!!!!!!")
    print("We also recommend you to use their original codes at https://github.com/YtongXie/UniMiSS-code/tree/main/UniMiSS/Downstream")
    print("The implementation here is without deep-supervision for fair comparisons with other methods !")
    return model


def DoDNet(args, pretrained_path='supervised_dodnet_unet_920.pth'):
    # CVPR 2021
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    model = DoDNet_UNet3D(num_classes=args.out_channels)
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using DoDNet-CVPR21 pretrained backbone weights !!!!!!!")
    return model


def stunet(args, pretrained_path='large_ep4k.model'):
    # arxiv
    pretrained_path = os.path.join(args.pretrained_root, pretrained_path)
    from STUNet import STUNet
    model = STUNet(input_channels=args.in_channels, num_classes=args.out_channels)
    model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
    model = load(model, model_dict)
    print("Using stunet pretrained backbone weights !!!!!!!")
    return model


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
    #         print(k)

    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}

    model.load_state_dict(new_state_dict, strict=True)

    return model


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="VoCo models")
    parser.add_argument("--pretrained_root", default='/home/linshan/pretrained/', type=str, help="pretrained_root")
    parser.add_argument("--pretrained_path", default='model_B.pt', help="checkpoint name for voco")

    parser.add_argument("--feature_size", default=48, type=int, help="feature size")
    parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
    parser.add_argument("--out_channels", default=4, type=int, help="number of output channels")

    parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
    parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
    parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")

    parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
    parser.add_argument("--dropout_path_rate", default=0.0, type=float, help="drop path rate")
    parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")
    parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")

    args = parser.parse_args()
    model = VoCo(args)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    input = torch.rand(1, 1, 96, 96, 96)
    output = model(input)
    print(output.shape)

    from thop import profile
    import torch
    import torchvision.models as models

    flops, params = profile(model, inputs=(input,))
    gflops = flops / 1e9
    print(f"GFLOPS: {gflops}")


