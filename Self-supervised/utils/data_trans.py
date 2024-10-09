import math
import os
from copy import deepcopy
import numpy as np
import torch
import pickle
from monai import data, transforms
from monai.data import *
from monai.transforms import *
from torch.utils.data import DataLoader, ConcatDataset
from utils.voco_trans import VoCoAugmentation


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
        self.valid_length = len(indices[self.rank: self.total_size: self.num_replicas])

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
        indices = indices[self.rank: self.total_size: self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_abdomen_trans(args):
    base_trans = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        Spacingd(keys=["image", "label"], pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear", "nearest")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0,
            a_max=250.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        SpatialPadd(keys=["image", "label"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode='constant'),

        # random
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0),

        SpatialPadd(keys=["image"], spatial_size=(192, 192, 64), mode='constant'),
        RandSpatialCropd(keys="image", roi_size=(192, 192, 64), max_roi_size=None, random_center=True, random_size=False, lazy=False),
        VoCoAugmentation(args, aug=True)
    ]

    return base_trans


def get_abdomen_trans_without_label(args):
    base_trans = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0,
            a_max=250.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode='constant'),

        # random
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0),
        Resized(keys="image", mode="bilinear", align_corners=True, spatial_size=(192, 192, 64)),

        SpatialPadd(keys=["image"], spatial_size=(192, 192, 64), mode='constant'),
        RandSpatialCropd(keys="image", roi_size=(192, 192, 64), max_roi_size=None, random_center=True, random_size=False, lazy=False),
        VoCoAugmentation(args, aug=True)
    ]

    return base_trans


def get_chest_trans(args):
    base_trans = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.25, 1.25, 5.0),
                 mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-1000.0,
            a_max=500.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode='constant'),

        # random
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0),

        SpatialPadd(keys=["image"], spatial_size=(192, 192, 64), mode='constant'),
        RandSpatialCropd(keys="image", roi_size=(192, 192, 64), max_roi_size=None, random_center=True, random_size=False, lazy=False),
        VoCoAugmentation(args, aug=True)
    ]

    return base_trans


def get_headneck_trans(args):
    base_trans = [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 1.5),
                 mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=-175.0,
            a_max=250.0,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode='constant'),

        # random
        transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=0),
        Resized(keys="image", mode="bilinear", align_corners=True, spatial_size=(192, 192, 64)),

        SpatialPadd(keys=["image"], spatial_size=(192, 192, 64), mode='constant'),
        RandSpatialCropd(keys="image", roi_size=(192, 192, 64), max_roi_size=None, random_center=True, random_size=False, lazy=False),
        VoCoAugmentation(args, aug=True)
    ]

    return base_trans