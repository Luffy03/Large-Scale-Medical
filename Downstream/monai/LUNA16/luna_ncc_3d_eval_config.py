from testers import *
from models import models_dict
from networks import networks_dict
from datasets_3D import datasets_dict_3D
import argparse
import os

models_name = models_dict.keys()
datasets_name = datasets_dict_3D.keys()
networks_name = networks_dict.keys()
os.environ['CUDA_VISIBLE_DEVICES'] = '4, 6'

parser = argparse.ArgumentParser(description='3D_Unet Test')
# Add default argument

parser.add_argument('--gpu_ids', type=int,default=[0, 1], help="For example 0,1 to run on two GPUs")
parser.add_argument('--network',  type=str, default='unet_3d_dense', choices=networks_name,
                    help='Networks')
parser.add_argument('--init_weight_type',type=str, choices=["kaiming",'normal','xavier','orthogonal'],
                    default="kaiming", help=" model init mode")
parser.add_argument('--note', type=str, default='test',
                    help='model note ')

parser.add_argument('--eval_dataset', type=str, default='luna_ncc',choices=datasets_name,
                    help='val data')
parser.add_argument('--im_channel', type=int, default=1, help="input image channel")
parser.add_argument('--class_num', type=int, default=1, help="output feature channel")
parser.add_argument('--normalization', type=str, choices=['sigmoid', 'softmax', 'none'],
                    default='sigmoid', help="the normalization of output")
parser.add_argument('--val_batch', type=int, default=64, help="val_batch")
parser.add_argument('--num_workers', type=int, default=0, help="dataloader numworkers")
parser.add_argument('--model_path', type=str, default='/jhcnas1/wulinshan/checkpoints/luna_ncc/unet_3d_dense_Simple_luna_ncc_rcb_(order_xyz)/20230718-111855/model_best.pth')
parser.add_argument('--save_results_path', type=str, default='save_luna')


if __name__ == '__main__':
    args = parser.parse_args()
    Tester = Classification2CTester(args)
    Tester.test_all_cases()
