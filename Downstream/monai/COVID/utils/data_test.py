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

import math
import os
import pickle
import numpy as np
import torch
import itertools as it
from monai import data, transforms
from monai.data import *


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image"]),
            transforms.EnsureChannelFirstd(keys=["image"]),
            transforms.Orientationd(keys=["image"], axcodes="RAS"),
            transforms.Spacingd(
                keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z), mode=("bilinear")
            ),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=args.a_min, a_max=args.a_max, b_min=args.b_min, b_max=args.b_max, clip=True
            ),
            transforms.CropForegroundd(keys=["image"], source_key="image"),
        ]
    )

    datalist = load_decathlon_datalist(datalist_json, True, "training", base_dir=data_dir)

    ds = Dataset(data=datalist,transform=transform)

    sampler = None
    loader = data.DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        num_workers=args.workers,
        sampler=sampler,
        pin_memory=True,
    )

    return loader, transform
