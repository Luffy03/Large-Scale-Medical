# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
from functools import partial
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from dataset.dataloader_test import get_test_loader_Atlas
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
# from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import *
from monai.utils.enums import MetricReduction
from monai.handlers import StatsHandler, from_engine
import matplotlib.pyplot as plt
from utils.utils import *
from PIL import Image
from monai import data, transforms
from monai.data import *

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--test_data_path", default="./test_examples/AbdomenAtlasTest/", type=str, help="test_data_path")
parser.add_argument(
    "--save_prediction_path", default="./test_examples/AbdomenAtlasPredict/", type=str, help="test_prediction_path")
parser.add_argument(
    "--trained_pth", default="./runs/logs/model_val50_91.88.pt", type=str, help="trained checkpoint directory")

roi = 96
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=4, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=10, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=16, type=int, help="number of workers")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")


def main():
    args = parser.parse_args()

    test_loader, test_transforms = get_test_loader_Atlas(args)

    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        use_checkpoint=args.use_checkpoint,
        use_v2=True
    )
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dict = torch.load(args.trained_pth)["state_dict"]
    model.load_state_dict(model_dict, strict=True)
    model.eval()
    model.to(device)

    # enable cuDNN benchmark
    torch.backends.cudnn.benchmark = True

    post_transforms = Compose([EnsureTyped(keys=["pred"]),
                               Invertd(keys=["pred"],
                                       transform=test_transforms,
                                       orig_keys="image",
                                       meta_keys="pred_meta_dict",
                                       orig_meta_keys="image_meta_dict",
                                       meta_key_postfix="meta_dict",
                                       nearest_interp=True,
                                       to_tensor=True),
                               AsDiscreted(keys="pred", argmax=False, to_onehot=args.out_channels),
                               ])

    with torch.no_grad():
        for idx, batch_data in enumerate(test_loader):
            torch.cuda.empty_cache()
            # data = batch_data["image"].cuda()

            data = batch_data["image"]
            data = data.cuda()

            with autocast(enabled=True):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

            batch_data['pred'] = logits.argmax(1)
            batch_data = post_transforms(batch_data)

            save_pred_dir = os.path.join(args.save_prediction_path, batch_data['name'][0], 'predictions')
            check_dir(save_pred_dir)

            organ_ls = ["aorta", "gall_bladder", "kidney_left", "kidney_right", "liver", "pancreas", "postcava",
                        "spleen", "stomach"]

            for idx, organ_name in enumerate(organ_ls):
                organ = batch_data['pred'][idx+1, :, :, :]
                batch_data['organ'] = organ
                save_transforms = Compose([SaveImaged(keys="organ", meta_keys="pred_meta_dict", output_dir=save_pred_dir,
                                separate_folder=False, folder_layout=None, output_postfix=organ_name,
                                resample=False)])
                save_transforms(batch_data)
                os.rename(os.path.join(save_pred_dir, 'ct_'+organ_name+'.nii.gz'), os.path.join(save_pred_dir, organ_name+'.nii.gz'))


if __name__ == "__main__":
    main()