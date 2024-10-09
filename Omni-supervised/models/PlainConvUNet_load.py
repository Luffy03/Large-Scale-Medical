from dynamic_network_architectures.architectures.unet import ResidualEncoderUNet, PlainConvUNet
from dynamic_network_architectures.building_blocks.helper import get_matching_instancenorm, convert_dim_to_conv_op
from torch import nn
import torch


def get_Plain_nnUNet(num_input_channels=1, num_classes=21, deep_supervision=False):
    UNet_base_num_features = 32
    unet_max_num_features = 320

    conv_kernel_sizes=[[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3],[3,3,3]]
    pool_op_kernel_sizes = [[1,1,1],[2,2,2],[2,2,2],[2,2,2],[2,2,2],[2,2,2]]
    n_conv_per_stage_encoder = [2,2,2,2,2,2]
    n_conv_per_stage_decoder = [2,2,2,2,2]

    dim = len(conv_kernel_sizes[0])
    conv_op = convert_dim_to_conv_op(dim)
    num_stages = len(conv_kernel_sizes)

    conv_or_blocks_per_stage = {
                'n_conv_per_stage': n_conv_per_stage_encoder,
                'n_conv_per_stage_decoder': n_conv_per_stage_decoder
            }
    kwargs = {
                'PlainConvUNet': {
                    'conv_bias': True,
                    'norm_op': get_matching_instancenorm(conv_op),
                    'norm_op_kwargs': {'eps': 1e-5, 'affine': True},
                    'dropout_op': None, 'dropout_op_kwargs': None,
                    'nonlin': nn.LeakyReLU, 'nonlin_kwargs': {'inplace': True},
                },
            }

    model = PlainConvUNet(
                input_channels=num_input_channels,
                n_stages=num_stages,
                features_per_stage=[min(UNet_base_num_features * 2 ** i,
                                        unet_max_num_features) for i in range(num_stages)],
                conv_op=conv_op,
                kernel_sizes=conv_kernel_sizes,
                strides=pool_op_kernel_sizes,
                num_classes=num_classes,
                deep_supervision=deep_supervision,
                **conv_or_blocks_per_stage,
                **kwargs['PlainConvUNet']
            )
    return model


if __name__ == '__main__':
    num_input_channels = 1
    model = get_Plain_nnUNet(num_input_channels)

    try:
        model_dict = torch.load("./checkpoint_final.pth", map_location=torch.device('cpu'))['network_weights']
        print(model_dict.keys())

        current_model_dict = model.state_dict()
        new_state_dict = {k: v if v.size() == current_model_dict[k].size() else current_model_dict[k] for k, v in
                          zip(current_model_dict.keys(), model_dict.values())}
        print(new_state_dict.keys())

        model.load_state_dict(new_state_dict, strict=True)
        print("Using pretrained nnUNet weights !")

        # torch.save(model_dict, 'model.pth')
        # torch.save(new_state_dict, 'new_model.pth')

    except ValueError:
        raise ValueError("Self-supervised pre-trained weights not available")

    x = torch.rand([1, num_input_channels, 96, 96, 96])
    y = model(x)
    print(y[0].shape)
