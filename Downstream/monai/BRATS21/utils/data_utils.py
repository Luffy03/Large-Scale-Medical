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

import json
import math
import os

import numpy as np
import torch

from monai import data, transforms
from monai.data import *
from monai.apps import DecathlonDataset
from copy import deepcopy
from monai.transforms import *
import pickle


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val


def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_list, val_list = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)

    train_transform = Compose(
        [
            # load 4 Nifti images and stack them together
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),

            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                        mode="constant"),

            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.sw_batch_size,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.1, spatial_axis=2),

            Convert_brats_Classesd(keys=["label"])
        ]
    )
    val_transform = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z]
            ),

            SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                        mode="constant"),

            RandShiftIntensityd(keys="image", offsets=0.1, prob=0),
            Convert_brats_Classesd(keys=["label"])
        ]
    )

    if args.use_persistent_dataset:
        print('use persistent')
        train_ds = PersistentDataset(data=train_list,
                                     transform=train_transform,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir=args.cache_dir)
    else:
        train_ds = data.Dataset(
            data=train_list, transform=train_transform)
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    if args.use_persistent_dataset:
        val_ds = PersistentDataset(data=val_list,
                                   transform=val_transform,
                                   pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                   cache_dir=args.cache_dir)
    else:
        val_ds = data.Dataset(data=val_list, transform=val_transform)

    val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False,
                            sampler=val_sampler, pin_memory=True,
                            persistent_workers=True,
                            num_workers=args.workers)
    return train_loader, val_loader


# class ConvertToMultiChannelBasedOnBratsClasses(Transform):
#     """
#     Convert labels to multi channels based on brats18 classes:
#     label 1 is the necrotic and non-enhancing tumor core
#     label 2 is the peritumoral edema
#     label 4 is the GD-enhancing tumor
#     The possible classes are TC (Tumor core), WT (Whole tumor)
#     and ET (Enhancing tumor).
#     """
#
#     backend = [TransformBackends.TORCH, TransformBackends.NUMPY]
#
#     def __call__(self, img: NdarrayOrTensor) -> NdarrayOrTensor:
#         # if img has channel dim, squeeze it
#         if img.ndim == 4 and img.shape[0] == 1:
#             img = img.squeeze(0)
#
#         result = [(img == 1) | (img == 4), (img == 1) | (img == 4) | (img == 2), img == 4]
#         # merge labels 1 (tumor non-enh) and 4 (tumor enh) and 2 (large edema) to WT
#         # label 4 is ET
#         return torch.stack(result, dim=0) if isinstance(img, torch.Tensor) else np.stack(result, axis=0)


class Convert_brats_Classesd(MapTransform):
    """
        Convert labels to multi channels based on brats18 classes:
        label 1 is the necrotic and non-enhancing tumor core
        label 2 is the peritumoral edema
        label 4 is the GD-enhancing tumor
        The possible classes are TC (Tumor core), WT (Whole tumor)
        and ET (Enhancing tumor).
        """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:

            new_img = d[key].clone()
            new_img[d[key] > 0] = 0

            new_img[d[key] == 1] = 1
            new_img[d[key] == 2] = 2
            new_img[d[key] == 4] = 3

            # print(torch.unique(d[key]), torch.unique(new_img.float()))
            d[key] = new_img.float()


        return d