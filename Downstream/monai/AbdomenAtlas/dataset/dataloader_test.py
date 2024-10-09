from monai.transforms import *

import sys
import nibabel as nib
import os
import torch
import numpy as np
from typing import Optional, Union
import math
import pickle
from monai.data import *
from monai.data import DataLoader, Dataset, list_data_collate, DistributedSampler, CacheDataset, SmartCacheDataset
from monai.config import DtypeLike, KeysCollection
from monai.transforms.transform import MapTransform
from monai.transforms.io.array import LoadImage
from monai.utils import ensure_tuple, ensure_tuple_rep
from monai.data.image_reader import ImageReader
from monai.utils.enums import PostFix
from utils.data_trans import *


DEFAULT_POST_FIX = PostFix.meta()

# class map for the AbdomenAtlas 1.0 dataset
class_map_abdomenatlas_1_0 = {
    0: "aorta",
    1: "gall_bladder",
    2: "kidney_left",
    3: "kidney_right",
    4: "liver",
    5: "pancreas",
    6: "postcava",
    7: "spleen",
    8: "stomach",
}

# class map for the AbdomenAtlas 1.1 dataset
class_map_abdomenatlas_1_1 = {
    0: 'aorta',
    1: 'gall_bladder',
    2: 'kidney_left',
    3: 'kidney_right',
    4: 'liver',
    5: 'pancreas',
    6: 'postcava',
    7: 'spleen',
    8: 'stomach',
    9: 'adrenal_gland_left',
    10: 'adrenal_gland_right',
    11: 'bladder',
    12: 'celiac_truck',
    13: 'colon',
    14: 'duodenum',
    15: 'esophagus',
    16: 'femur_left',
    17: 'femur_right',
    18: 'hepatic_vessel',
    19: 'intestine',
    20: 'lung_left',
    21: 'lung_right',
    22: 'portal_vein_and_splenic_vein',
    23: 'prostate',
    24: 'rectum'
}


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


class LoadSelectedImaged(MapTransform):
    """
    Custom transform to load a specific image and metadata using a flexible reader.

    Args:
        keys: Keys of the data dictionary to load selected images.
        reader: Image reader object or string reference.
        dtype: Data type for loaded images.
        meta_keys: Keys to store metadata along with image data.
        meta_key_postfix: Suffix for metadata keys.
        overwriting: Flag to allow overwriting existing metadata.
        image_only: Load only the image data (not metadata).
        ensure_channel_first: Reshape image into channel-first format if necessary.
        simple_keys: Use simplified, top-level data keys.
        allow_missing_keys: If True, missing data keys are ignored
    """

    def __init__(
            self,
            keys: KeysCollection,
            reader: Optional[Union[ImageReader, str]] = None,
            dtype: DtypeLike = np.float32,
            meta_keys: Optional[KeysCollection] = None,
            meta_key_postfix: str = DEFAULT_POST_FIX,
            overwriting: bool = False,
            image_only: bool = False,
            ensure_channel_first: bool = False,
            simple_keys: bool = False,
            allow_missing_keys: bool = False,
            *args,
            **kwargs,
    ) -> None:
        super().__init__(keys, allow_missing_keys)
        self._loader = LoadImage(reader, image_only, dtype, ensure_channel_first, simple_keys, *args, **kwargs)
        if not isinstance(meta_key_postfix, str):
            raise TypeError(f"meta_key_postfix must be a str but is {type(meta_key_postfix).__name__}.")
        self.meta_keys = ensure_tuple_rep(None, len(self.keys)) if meta_keys is None else ensure_tuple(meta_keys)
        if len(self.keys) != len(self.meta_keys):
            raise ValueError("meta_keys should have the same length as keys.")
        self.meta_key_postfix = ensure_tuple_rep(meta_key_postfix, len(self.keys))
        self.overwriting = overwriting

    def register(self, reader: ImageReader):
        self._loader.register(reader)

    def __call__(self, data, reader: Optional[ImageReader] = None):
        d = dict(data)
        for key, meta_key, meta_key_postfix in self.key_iterator(d, self.meta_keys, self.meta_key_postfix):
            data = self._loader(d[key], reader)
            if self._loader.image_only:
                d[key] = data
            else:
                if not isinstance(data, (tuple, list)):
                    raise ValueError("loader must return a tuple or list (because image_only=False was used).")
                d[key] = data[0]
                if not isinstance(data[1], dict):
                    raise ValueError("metadata must be a dict.")
                meta_key = meta_key or f"{key}_{meta_key_postfix}"
                if meta_key in d and not self.overwriting:
                    raise KeyError(f"Metadata with key {meta_key} already exists and overwriting=False.")
                d[meta_key] = data[1]

        return d


def get_test_loader_Atlas(args):
    """
    Creates training transforms, constructs a dataset, and returns a dataloader.

    Args:
        args: Command line arguments containing dataset paths and hyperparameters.
    """
    test_transforms = transforms.Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(args.space_x, args.space_y, args.space_z),
                 mode=("bilinear")),
        ScaleIntensityRanged(
            keys=["image"],
            a_min=args.a_min,
            a_max=args.a_max,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),
        CropForegroundd(keys=["image"], source_key="image"),
        SpatialPadd(keys=["image"], spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                    mode='constant'),
    ])

    # constructing training dataset
    test_img = []
    test_name = []

    dataset_list = os.listdir(args.test_data_path)

    for item in dataset_list:
        name = item
        test_img_path = os.path.join(args.test_data_path, name, 'ct.nii.gz')
        test_img.append(test_img_path)
        test_name.append(name)

    data_dicts_test = [{'image': image, 'name': name}
                        for image, name in zip(test_img, test_name)]

    print('test len {}'.format(len(data_dicts_test)))

    test_ds = Dataset(data=data_dicts_test, transform=test_transforms)
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=None, pin_memory=True
    )
    return test_loader, test_transforms
