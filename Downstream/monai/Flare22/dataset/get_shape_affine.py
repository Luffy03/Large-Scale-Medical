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
from utils.utils import dice, resample_3d
from utils.utils import AverageMeter, distributed_all_gather

from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction

import resource
from tqdm import tqdm
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))
print('Setting resource limit:', str(resource.getrlimit(resource.RLIMIT_NOFILE)))


def main():

    raw_path = '/data/linshan/CTs/Flare22/imagesTs/'
    shape = dict()
    affine = dict()
    img_names = os.listdir(raw_path)
    for img_name in tqdm(img_names):
        img_path = os.path.join(raw_path, img_name)
        nii_img = nib.load(img_path)

        shape[img_name] = nii_img.get_fdata().shape
        affine[img_name] = nii_img.affine

    data = {'shape': shape, 'affine': affine}
    np.save('raw_data.npy', data)


if __name__ == "__main__":
    main()