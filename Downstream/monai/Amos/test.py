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
from utils.data_test import get_loader
import SimpleITK as sitk
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import *
from monai.utils.enums import MetricReduction
from monai.handlers import StatsHandler, from_engine
import matplotlib.pyplot as plt
from utils.utils import *
from PIL import Image

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))
from monai import data, transforms
from monai.data import *

os.environ['CUDA_VISIBLE_DEVICES'] = "4"
os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '28890'

parser = argparse.ArgumentParser(description="Swin UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./runs/logs_swin_base_voco160k/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="/data/linshan/CTs/Amos2022/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_CT.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name",
    default="model_final.pt",
    type=str,
    help="pretrained model name",
)
roi = 96
parser.add_argument("--use_normal_dataset", default=True, help="use monai Dataset class")
parser.add_argument("--feature_size", default=48, type=int, help="feature size")
parser.add_argument("--batch_size", default=1, type=int, help="number of batch size")
parser.add_argument("--sw_batch_size", default=2, type=int, help="number of sliding window batch size")
parser.add_argument("--infer_overlap", default=0.75, type=float, help="sliding window inference overlap")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=16, type=int, help="number of output channels")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=1.5, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=roi, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=roi, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=roi, type=int, help="roi size in z direction")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--spatial_dims", default=3, type=int, help="spatial dimension of input data")
parser.add_argument("--use_checkpoint", default=True, help="use gradient checkpointing to save memory")


def main():
    args = parser.parse_args()
    args.test_mode = True
    output_directory = "./pred/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    val_loader, test_transforms = get_loader(args)

    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
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

    model_dict = torch.load(pretrained_pth)["state_dict"]
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
                                     nearest_interp=False,
                                     to_tensor=True),
                               AsDiscreted(keys="pred", argmax=True, to_onehot=None),
                               SaveImaged(keys="pred", meta_keys="pred_meta_dict", output_dir=output_directory,
                                          separate_folder=False, folder_layout=None,
                                          resample=False),
                               ])

    cmap = color_map()

    num = 0

    with torch.no_grad():
        for idx, batch_data in enumerate(val_loader):
            torch.cuda.empty_cache()

            # img_name = batch_data["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            # print('img_name:', img_name, num)
            # num += 1

            if isinstance(batch_data, list):
                data = batch_data
            else:
                data = batch_data["image"].cuda()

            with autocast(enabled=True):
                if model_inferer is not None:
                    logits = model_inferer(data)
                else:
                    logits = model(data)

            batch_data['pred'] = logits

            # ori = torch.argmax(logits, 1).cpu().numpy().astype(np.uint8)[0]
            # ori = Image.fromarray(ori[:, :, 50].astype(np.uint8), mode='P')
            # ori.putpalette(cmap)

            batch_data = [post_transforms(i) for i in
                         decollate_batch(batch_data)]  # apply post-processing to output tensors

            # test_img, val_outputs = from_engine(["image", "pred"])(batch_data)

            # test_img = test_img[0][0].data.cpu().numpy()
            # print(test_img.shape)

            # c = val_outputs[0].shape[-1]
            # val_outputs = val_outputs[0].argmax(0).cpu().numpy().astype(np.uint8)

            # # # vis
            # print(np.unique(val_outputs[:, :, c//3].astype(np.uint8)))
            # val = Image.fromarray(val_outputs[:, :, c//3].astype(np.uint8), mode='P')
            # val.putpalette(cmap)

            # # # show
            # plt.figure("check", (18, 6))
            # plt.subplot(1, 2, 1)
            # plt.imshow(test_img[:, :, c//3], cmap="gray")
            # # plt.imshow(ori)
            #
            # plt.subplot(1, 2, 2)
            # plt.imshow(val)
            # plt.show()

            # # # save predict
            # seg = sitk.GetImageFromArray(val_outputs)
            # seg.SetSpacing(img_itk.GetSpacing())
            # seg.SetDirection(img_itk.GetDirection())
            # sitk.WriteImage(seg, os.path.join(output_directory, img_name[:-12] + '.nii.gz'))


if __name__ == "__main__":
    main()