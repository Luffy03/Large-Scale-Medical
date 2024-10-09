import argparse
from torch.nn import init
from collections.abc import Iterable
from networks.unet import UNet2D, U_Net_Encoder, UNet2D_Dense, UNet2D_RPL, UNet2D_JigSaw
from networks.unet3d import UNet3D, UNet3D_Dense, UNet3D_Encoder, UNet3D_wo_skip, UNet3D_RPL, UNet3D_JigSaw, UNet3D_RKB, UNet3D_RKBP
from networks.PCRL import PCRLModel
from networks.PCRL3d import PCRLModel3d
from networks.MyPCLR3d import PCRLEncoder3d, PCRLDecoder3d, PCRLDecoder3d_wo_skip
from networks.MyPCRL2d import PCRLEncoder2d, PCRLDecoder2d

from networks.swinunetr import Swin

networks_dict= {
    'unet_2d': UNet2D,
    'unet_2d_dense': UNet2D_Dense,
    'unet_2d_rpl': UNet2D_RPL,
    'unet_2d_jigsaw':UNet2D_JigSaw,
    'unet_3d': UNet3D,
    'unet_3d_wo_skip': UNet3D_wo_skip,
    'unet_encoder': U_Net_Encoder,
    'unet_3d_encoder': UNet3D_Encoder,
    'unet_3d_dense': UNet3D_Dense,
    'unet_3d_rpl': UNet3D_RPL,
    'unet_3d_jigsaw': UNet3D_JigSaw,
    'unet_3d_rkb': UNet3D_RKB,
    'unet_3d_rkbp': UNet3D_RKBP,
    'pcrl': PCRLModel,
    'pcrl_3d': PCRLModel3d,
    'pcrl_3d_encoder': PCRLEncoder3d,
    'pcrl_3d_decoder': PCRLDecoder3d,
    'pcrl_2d_encoder': PCRLEncoder2d,
    'pcrl_2d_decoder': PCRLDecoder2d,

    '3d_swin': Swin,
}


def init_weights(net, init_type='kaiming', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)
    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_networks(args):
    network_name = args.network
    if network_name == "unet_3d" or network_name == 'unet_3d_wo_skip':
        network = networks_dict[network_name](args.im_channel, args.class_num, args.normalization)

    elif network_name == 'unet_3d_encoder' or network_name == 'unet_3d_encoder_avgpool':
        network = networks_dict[network_name](args.im_channel)

    elif network_name == 'unet_encoder':
        print(network_name)
        network = networks_dict[network_name](args.im_channel, args.projection_dim)

    elif network_name == 'pcrl_3d' or network_name == 'pcrl_3d_encoder' or network_name == 'pcrl_3d_decoder'  or network_name == 'pcrl_2d_encoder' or network_name == 'pcrl_2d_decoder':
        print('student:', args.is_student)
        network = networks_dict[network_name](in_channels=args.im_channel, n_class=args.class_num,
                                              student=args.is_student, norm ='bn')

    elif network_name == 'unet_3d_jigsaw' or network_name == 'unet_3d_rkb'  or network_name == 'unet_3d_rkbp':
        network = networks_dict[network_name](in_channels=args.im_channel, order_n_class=args.order_class_num, num_cubes=args.num_grids_per_axis ** 3)
        
    elif network_name == 'unet_2d_jigsaw':
        network = networks_dict[network_name](im_ch=args.im_channel, output_ch=args.order_class_num,
                                              num_cubes=args.num_grids_per_axis ** 2)
    else:
        network = networks_dict[network_name](args.im_channel, args.class_num, args.normalization)

    ##raise  NotImplementedError("the model is not exists !")
    init_weights(network, args.init_weight_type)
    return network


def set_freeze_by_keywords(model, keywords, freeze=True):
    for k, v in model.named_parameters():
        v.requires_grad = True
        if any(k.find(x) != -1 for x in keywords):
            print('changing %s' % k)
            v.requires_grad = not freeze


def freeze_by_keywords(model, keywords):
    print('****** freezing ******')
    set_freeze_by_keywords(model, keywords, True)


def unfreeze_by_keywords(model, keywords):
    print('****** unfreezing ******')
    set_freeze_by_keywords(model, keywords, False)


def set_freeze_by_names(model, layer_names, freeze=True):
    if not isinstance(layer_names, Iterable):
        layer_names = [layer_names]
    for name, child in model.named_children():
        if name not in layer_names:
            continue
        for param in child.parameters():
            # print(param.name)
            param.requires_grad = not freeze


def freeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, True)


def unfreeze_by_names(model, layer_names):
    set_freeze_by_names(model, layer_names, False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.im_channel = 3
    args.class_num = 3
    args.init_weight_type= "kaiming"
    name_list = list(networks_dict.keys())
    for i in range(len(name_list)):
        args.network= name_list[i]
        model = get_networks(args)
        #print(model)
