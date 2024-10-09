from models.models import *


def get_model(args):
    if args.name == 'VoCo':
        return VoCo(args)
    elif args.name == 'suprem':
        return SuPrem(args)
    elif args.name == 'swin':
        return Swin(args)
    elif args.name == 'clip_driven':
        return Universal(args)
    elif args.name == 'mg':
        return MG(args)
    elif args.name == 'unimiss':
        return UniMiss(args)
    elif args.name == 'dodnet':
        return DoDNet(args)
    else:
        print('Without pre-training !')
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
        return model