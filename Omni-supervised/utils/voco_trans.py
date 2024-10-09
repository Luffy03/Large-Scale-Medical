from collections.abc import Callable, Sequence
from torch.utils.data import Dataset as _TorchDataset
from torch.utils.data import Subset
import collections
import numpy as np
from monai.data import *
import pickle
from monai.transforms import *
from math import *


def threshold(x):
    # threshold at 0
    return x > 0.3


class VoCoAugmentation():
    def __init__(self, args, aug):
        self.args = args
        self.aug = aug

    def __call__(self, x_in):
        if 'label' in x_in.keys():
            del x_in['label']

        num_crops = 3
        max_roi = 192

        crops_trans = get_crop_transform(num_crops=num_crops, aug=self.aug)

        vanilla_trans, labels = get_vanilla_transform(num=self.args.sw_batch_size, num_crops=num_crops,
                                                      max_roi=max_roi, aug=self.aug)

        imgs = []
        for trans in vanilla_trans:
            img = trans(x_in)
            imgs.append(img)

        crops = []
        for trans in crops_trans:
            crop = trans(x_in)
            crops.append(crop)

        return imgs, labels, crops


def get_vanilla_transform(num=2, num_crops=4, roi=64, max_roi=256, aug=False):
    vanilla_trans = []
    labels = []
    for i in range(num):
        center_x, center_y, label = get_position_label(roi=roi, base_roi=roi,
                                                       max_roi=max_roi,
                                                       num_crops=num_crops)
        if aug:
            trans = Compose([
                SpatialCropd(keys=['image'],
                             roi_center=[center_x, center_y, roi // 2],
                             roi_size=[roi, roi, roi]),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                ])
        else:
            trans = Compose([
                SpatialCropd(keys=['image'],
                             roi_center=[center_x, center_y, roi // 2],
                             roi_size=[roi, roi, roi]),
                ])

        vanilla_trans.append(trans)
        labels.append(label)

    labels = np.concatenate(labels, 0).reshape(num, num_crops * num_crops)

    return vanilla_trans, labels


def get_crop_transform(num_crops=4, roi=64, aug=True):
    voco_trans = []
    # not symmetric at axis x !!!
    for i in range(num_crops):
        for j in range(num_crops):
            center_x = (i + 1 / 2) * roi
            center_y = (j + 1 / 2) * roi
            center_z = roi // 2

            if aug:
                trans = Compose([
                    SpatialCropd(keys=['image'],
                                 roi_center=[center_x, center_y, center_z],
                                 roi_size=[roi, roi, roi]),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=0),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=1),
                    RandFlipd(keys=["image"], prob=0.2, spatial_axis=2),
                    RandRotate90d(keys=["image"], prob=0.2, max_k=3),
                    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
                    ]
                )
            else:
                trans = Compose([
                    SpatialCropd(keys=['image'],
                                 roi_center=[center_x, center_y, center_z],
                                 roi_size=[roi, roi, roi]),
                    ],
                )

            voco_trans.append(trans)

    return voco_trans


def get_position_label(roi=64, base_roi=64, max_roi=256, num_crops=4):
    assert roi*num_crops == max_roi, print('roi, num_crops, max_roi:',roi, num_crops, max_roi)

    half = roi // 2
    center_x, center_y = np.random.randint(low=half, high=max_roi - half), \
        np.random.randint(low=half, high=max_roi - half)
    # center_x, center_y = np.random.randint(low=half, high=half+1), \
    #     np.random.randint(low=half, high=half+1)
    # center_x, center_y = roi + half, roi + half
    # print(center_x, center_y)

    x_min, x_max = center_x - half, center_x + half
    y_min, y_max = center_y - half, center_y + half

    total_area = roi * roi
    labels = []
    for i in range(num_crops):
        for j in range(num_crops):
            crop_x_min, crop_x_max = i * base_roi, (i + 1) * base_roi
            crop_y_min, crop_y_max = j * base_roi, (j + 1) * base_roi

            dx = min(crop_x_max, x_max) - max(crop_x_min, x_min)
            dy = min(crop_y_max, y_max) - max(crop_y_min, y_min)
            if dx <= 0 or dy <= 0:
                area = 0
            else:
                area = (dx * dy) / total_area
            labels.append(area)

    labels = np.asarray(labels).reshape(1, num_crops * num_crops)

    return center_x, center_y, labels


if __name__ == '__main__':
    center_x, center_y, labels = get_position_label(roi=64, base_roi=64, max_roi=192, num_crops=3)
    print(center_x, center_y, labels)