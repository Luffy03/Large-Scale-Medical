import torchvision.transforms.functional as tf
import torch
from torchvision import transforms
import numpy as np
from scipy import ndimage
import random
import os
from datasets_3D.paths import Path
import SimpleITK as sitk
from .base_seg import SegmentationBaseTrainset
from monai.transforms import AddChannel, Compose, RandAffined, RandRotated, RandRotate90d, RandFlipd, apply_transform, ToTensor


class SegmentationLunaSet(SegmentationBaseTrainset):
    """
    Training/Test dataset for segmentation in the LUNA dataset (NCS).
    NCS: segment the lung nodule in each proposal cube, which contains ROI.
    """
    def __init__(self,
                 config,
                 base_dir,
                 flag='train',
                 ):
        super(SegmentationLunaSet, self).__init__(config, base_dir, flag)
        self.flag = flag
        self.config = config
        self.crop_size = config.input_size
        self.num_classes = config.class_num
        self._base_dir = base_dir
        # load data
        self.all_images, self.all_masks = self.load_image(data_path=self._base_dir, status=self.flag)

        assert self.all_images.shape == self.all_masks.shape
        assert len(self.all_images) != 0, "the images can`t be zero!"

        ### Display status
        print('Number of images in {}: {:d}'.format(flag, self.all_images.shape[0]))

        # get aug transforms
        self.aug_transforms = self.get_aug_transforms()

    def __len__(self):
            return self.all_images.shape[0]

    def load_image(self, data_path, status=None):
        x = np.squeeze(np.load(os.path.join(data_path, 'x_' + status + '_64x64x32.npy')))
        y = np.squeeze(np.load(os.path.join(data_path, 'm_' + status + '_64x64x32.npy')))
        x = np.expand_dims(x, axis=1)
        y = np.expand_dims(y, axis=1)
        return x, y

    def __getitem__(self, index):
        # (N, C, 64, 64, 32)
        image, label = self.all_images[index], self.all_masks[index]

        # aug
        if self.flag == 'train':
            sample_dict = {'image': image, 'label': label}
            sample_dict = self.aug_transforms(sample_dict)
            image = sample_dict['image']
            label = sample_dict['label']

        return torch.from_numpy(image.astype(np.float32)), torch.from_numpy(label.astype(np.int32)), index

    def get_aug_transforms(self):
        train_transforms = Compose(
            [# AddChannel(keys=["image", "label"]), # add this if the data has no channel.
             RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(0, 1)),
             RandRotated( keys=["image", "label"], mode=["bilinear", "nearest"], prob=0.6, range_x=20, range_y=20, range_z=0),
             RandRotate90d(keys=["image", "label"], prob=0.5, spatial_axes=(0, 1))])

        return train_transforms

    def __str__(self):
       pass




