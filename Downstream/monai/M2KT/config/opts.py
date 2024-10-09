from __future__ import print_function
import argparse
import os
import json
from datetime import datetime, timedelta
from misc.utils import set_logging
import logging


def parse_opt(prefix=None):
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='/home/xuefeng/Chest_New_nitfy_2/',
                        help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='/home/xuefeng/label/chest/chest_ctrg_r2gen_nitfy_ori_split.json',
                        help='the path to the directory containing the data.')
    parser.add_argument('--label_path', type=str, default='/home/xuefeng/M2KT/data_csv/CTRG_finding_labels.csv',
                        help='the path to the directory containing the label.')
    parser.add_argument('--image_size', type=int, default=96, help='')
    parser.add_argument('--crop_size', type=int, default=96, help='')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='ctrg',
                        choices=['iu_xray', 'mimic_cxr', 'covid', 'covidall', 'ctrg'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=150, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=8, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=4, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('-v', '--version', type=str, default="VoCo", help='main model version')
    parser.add_argument('--visual_extractor', type=str, default='swin3d',
        choices=['resnet101', 'swin3d'],
                        help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True,
                        help='whether to load the pretrained visual extractor')
    parser.add_argument('--pretrain_cnn_file', type=str, default='', help='the visual extractor to be used.')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=512, help='for densenet = 1024, for efficientnet = 1280')
    parser.add_argument('--feature_size', type=int, default=48, help='for swinunetr')

    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_memory_heads', type=int, default=8, help='the number of heads in memory.')
    parser.add_argument('--num_layers', type=int, default=1, help='the number of layers of Transformer.')
    parser.add_argument('--num_slices', type=int, default=2, help='the number of slices of the image.')
    parser.add_argument('--num_labels', type=int, default=14, help='the number of labels.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.1, help='the dropout rate of the output layer.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search',
                        help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='results/', help='the patch to save the models.')
    parser.add_argument('--expe_name', type=str, default='swin3d_96_nii_scratch', help='extra experiment name')
    parser.add_argument('--record_dir', type=str, default='records/',
                        help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'],
                        help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--grad_clip', type=float, default=5)
    parser.add_argument('--test_steps', type=int, default=0, help='the period of test in training')

    # Optimization
    parser.add_argument('--label_loss', default=False, action='store_true')
    parser.add_argument('--rank_loss', default=False, action='store_true')
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=20, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--gpu', type=str, default='2', help='')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    # for Relational Memory
    parser.add_argument('--num_slots', type=int, default=60, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # config
    parser.add_argument('--cfg', type=str, default='/home/xuefeng/M2KT/config/ctrg_swin3d.yml',
                    help='configuration; similar to what is used in detectron')
    parser.add_argument(
        '--set_cfgs', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]\n This has higher priority'
             'than cfg file but lower than other args. (You can only overwrite'
             'arguments that have alerady been defined in config file.)',
        default=[], nargs='+')
    # How will config be used
    # 1) read cfg argument, and load the cfg file if it's not None
    # 2) Overwrite cfg argument with set_cfgs
    # 3) parse config argument to args.
    # 4) in the end, parse command line argument and overwrite args

    # step 1: read cfg_fn
    args = parser.parse_args()
    if args.cfg is not None or args.set_cfgs is not None:
        from .config import CfgNode
        if args.cfg is not None:
            cn = CfgNode(CfgNode.load_yaml_with_base(args.cfg))
        else:
            cn = CfgNode()
        if args.set_cfgs is not None:
            cn.merge_from_list(args.set_cfgs)
        for k,v in cn.items():
            if not hasattr(args, k):
                print('Warning: key %s not in args' % k)
            setattr(args, k, v)
        args = parser.parse_args(namespace=args)

    # Check if args are valid
    assert args.d_model > 0, "rnn_size should be greater than 0"
    assert args.num_layers > 0, "num_layers should be greater than 0"
    assert args.d_vf > 0, "input_encoding_size should be greater than 0"
    assert args.batch_size > 0, "batch_size should be greater than 0"
    assert args.drop_prob_lm >= 0 and args.drop_prob_lm < 1, "drop_prob_lm should be between 0 and 1"
    assert args.beam_size > 0, "beam_size should be greater than 0"

    # add training parameters to save dir
    if args.resume == None or args.expe_name != '':
        expe_name = f"V{args.version}_" \
                    + args.visual_extractor \
                    + ("_labelloss" if args.label_loss else "") \
                    + ("_rankloss" if args.rank_loss else "") \
                    + ("_" + args.expe_name if args.expe_name != "" else "") \
                    + (datetime.now() + timedelta(hours=15)).strftime("_%Y%m%d-%H%M%S")
        expe_name = prefix + '_' + expe_name if prefix else expe_name
        args.save_dir = os.path.join(args.save_dir, args.dataset_name, expe_name)
    else:
        args.save_dir = os.path.split(args.resume)[0]
        expe_name = os.path.split(args.save_dir)[1]

    # Save config for reproduce
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(os.path.join(args.save_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f)

    set_logging(os.path.join(args.save_dir, f'{expe_name}.log'))
    logging.info(f'Logging Dir: {args.save_dir}')

    # modify visual feature projection dimension
    if args.visual_extractor == "efficientnet" and args.d_vf != 1280:
        args.d_vf = 1280
    elif args.visual_extractor == "densenet" and args.d_vf != 1024:
        args.d_vf = 1024
    elif args.visual_extractor == "resnet101" and args.d_vf != 2048:
        args.d_vf = 2048
    logging.info(f"Visual Extractor:{args.visual_extractor}   d_vf: {args.d_vf}")

    # # set gpu
    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # logging.info(f"==>> Set GPU devices: {args.gpu}")

    return args


def add_eval_options(parser):
    pass

def add_diversity_opts(parser):
    pass


# Sampling related options
def add_eval_sample_opts(parser):
    pass


if __name__ == '__main__':
    import sys
    sys.argv = [sys.argv[0]]
    args = parse_opt()
    print(args)
    print()
    sys.argv = [sys.argv[0], '--cfg', 'covid.yml']
    args1 = parse_opt()
    print(dict(set(vars(args1).items()) - set(vars(args).items())))
    print()
    sys.argv = [sys.argv[0], '--cfg', 'covid.yml', '--visual_extractor', 'densenet']
    args2 = parse_opt()
    print(dict(set(vars(args2).items()) - set(vars(args1).items())))