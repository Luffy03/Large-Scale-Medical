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
from collections.abc import Callable, Sequence
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import collections
import numpy as np
from monai.data import *
import pickle
from monai.transforms import *
from math import *
from torch.utils.data import DataLoader, ConcatDataset
from utils.data_trans import *
from utils.data_utils_abdomen import get_ds_abdomen
from utils.data_utils_headneck import get_ds_headneck
from utils.data_utils_chest import get_ds_chest


def get_loader(args):
    abdomen_ds = get_ds_abdomen(args)
    headneck_ds = get_ds_headneck(args)
    chest_ds = get_ds_chest(args)

    abdomen_ls = []
    for _ in range(8):
        abdomen_ls.append(abdomen_ds)
    abdomen_ds = ConcatDataset(abdomen_ls)

    headneck_ls = []
    for _ in range(8):
        headneck_ls.append(headneck_ds)
    headneck_ds = ConcatDataset(headneck_ls)

    train_ds = ConcatDataset(
        [abdomen_ds, headneck_ds,
         chest_ds
         ])

    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    return train_loader