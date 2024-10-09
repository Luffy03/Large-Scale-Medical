import torch
import numpy as np
import os
from .base_seg import SegmentationBaseTrainset, SegmentationBaseTestset
from monai.transforms import AddChannel, Compose, RandAffined, RandRotated, RandRotate90d, RandFlipd, apply_transform, ToTensor


class SegmentationLiTSTrainSet(SegmentationBaseTrainset):
    """
    Training dataset for segmentation in the LiTS dataset (LCS).
    LCS: segment the liver in each image.
    """
    def __init__(self,
                 config,
                 base_dir,
                 flag='train',
                 ):
        """
        :param base_dir: path to Pancreas dataset directory
        :param split: train/valid
        """

        super(SegmentationLiTSTrainSet, self).__init__(config, base_dir, flag)
        self.flag = flag
        self.config = config
        self.crop_size = config.input_size
        self.num_classes = config.class_num
        self._base_dir = base_dir + '/' + flag

        # load data
        self.all_images = os.listdir(self._base_dir + '/volume')
        self.all_masks = list(
            map(lambda x: x.replace('volume', 'segmentation'), self.all_images))

        self.all_images = list(map(lambda x: os.path.join(self._base_dir + '/volume', x), self.all_images))
        self.all_masks = list(map(lambda x: os.path.join(self._base_dir + '/segmentation', x), self.all_masks))


        assert len(self.all_images) == len(self.all_masks)

        assert len(self.all_images) != 0, "the images can`t be zero!"

        ### Display status
        print('Number of images in {}: {:d}'.format(flag, len(self.all_images)))

        # get aug transforms
        self.aug_transforms = self.get_aug_transforms()

    def __len__(self):
            return len(self.all_images)

    # def __getitem__(self, index):
    #     ### only for ROI image
    #     image_path, label_path = self.all_images[index], self.all_masks[index]
    #     image_index = os.path.split(image_path)[1][:-4]
    #
    #     image_array, label_array = self. _get_img_gt_pair(image_path, label_path)
    #
    #     # [c, d, h, w]
    #
    #     if self.flag == 'train':
    #         # randomly crop along z axis
    #         assert self.crop_size[1] == self.crop_size[2] == 256
    #         start_slice = random.randint(0, image_array.shape[-3] - self.crop_size[0])
    #         end_slice = start_slice + self.crop_size[0] - 1
    #
    #         image_array = image_array[:, start_slice:end_slice + 1, :, :]
    #         label_array = label_array[:, start_slice:end_slice + 1, :, :]
    #
    #     return torch.from_numpy(image_array).float(), torch.from_numpy(label_array).float(), image_index

    def random_crop_along_z_axis(self, image, label, fg_prob):
        """Crop the image in a sample randomly along z axis.
              Args:
                  image:[C, D, H, W]
                  label:[[K, D, H, W]
                  crop_size: the desired output size: [patch_D, H, W]
                  out_image:[C, patch_D, H, W]
                  out_label:[K, patch_D, H, W]
               """
        _, d, h, w = image.shape
        if np.random.uniform() > fg_prob:

            d1 = np.random.randint(0, d - self.crop_size[0])

            image = image[:,  d1:d1 + self.crop_size[0], :, :]
            label = label[:, d1:d1 + self.crop_size[0], :, :]
        else:
            # # Find the ROI
            z = np.any(label, axis=(-2, -1)).squeeze()
            start_slice, end_slice = np.where(z)[0][[0, -1]]
            print(start_slice, end_slice)

            # To prevent cropping out of border
            end_slice = min(end_slice, d - self.crop_size[0])
            start_slice = min(start_slice, end_slice)

            if start_slice >= end_slice:
                start_slice = end_slice-1

            d1 = np.random.randint(start_slice, end_slice)

            image = image[:, d1:d1 + self.crop_size[0], :, :]
            label = label[:, d1:d1 + self.crop_size[0], :, :]

        return image, label

    def __getitem__(self, index):

        image_path, label_path = self.all_images[index], self.all_masks[index]
        image_index = os.path.split(image_path)[1][:-4]

        image_array, label_array = self._get_img_gt_pair(image_path, label_path)

        if self.flag == 'train':
            # randomly crop along z axis
            assert self.crop_size[1] == self.crop_size[2] == 256
            image_array, label_array = self.random_crop_along_z_axis(image_array, label_array, fg_prob=0.3)
            # augmentations
            # sample_dict = {'image': image_array, 'label': label_array}
            # sample_dict = self.aug_transforms(sample_dict)
            # image_array = sample_dict['image']
            # label_array = sample_dict['label']

        return torch.from_numpy(image_array).float(), torch.from_numpy(label_array).float(), image_index

    def get_aug_transforms(self):
        # (num_channels, spatial_dim_1[, spatial_dim_2, ?]).monai.transforms
        # [y, z, x]
        train_transforms = Compose(
            [# AddChannel(keys=["image", "label"]), # add this if the data has no channel.
             RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=(1, 2))])
        return train_transforms

    def __str__(self):
       pass


class SegmentationLiTSTestset(SegmentationBaseTestset):
    """
    Test dataset for segmentation in the LiTS dataset (LCS).
    """
    def __init__(self, config, base_dir, flag='train'):
        super(SegmentationLiTSTestset, self).__init__(config, base_dir, flag)
        self.config = config
        self._base_dir = base_dir + '/' + flag
        self.all_images = []
        self.order = config.order
        # load data
        self.all_images = os.listdir(self._base_dir + '/volume')
        self.all_masks = list(
            map(lambda x: x.replace('volume', 'segmentation'), self.all_images))

        self.all_images = list(map(lambda x: os.path.join(self._base_dir + '/volume', x), self.all_images))
        self.all_masks = list(map(lambda x: os.path.join(self._base_dir + '/segmentation', x), self.all_masks))

        assert len(self.all_images) == len(self.all_masks)
        assert len(self.all_images) != 0, "the images can`t be zero!"

        ### Display status
        print('Number of images in {}: {:d}'.format(flag, len(self.all_images)))

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        image_path, label_path = self.all_images[index], self.all_masks[index]
        image_index = os.path.split(image_path)[1][:-4]

        image, label = self._get_img_gt_pair(image_path, label_path)

        if self.order == 'xyz':
            image = np.transpose(image, (0, 3, 2, 1))
            label = np.transpose(label, (0, 3, 2, 1))

        org_shape = np.array(label.shape)

        # padding
        full_image = self.padding_image(image, self.cut_params)
        new_shape = np.array(full_image.shape)

        # cut patches
        patches = self.extract_ordered_overlap(full_image, self.cut_params)

        # image info
        image_info = {'org_shape': torch.from_numpy(org_shape),
                      'new_shape': torch.from_numpy(new_shape),
                      'image_index': image_index}

        # patches: [N, C, pd, ph, pw], label [K, D, H, W]
        return torch.from_numpy(image),\
               torch.from_numpy(full_image),\
               torch.from_numpy(patches),\
               torch.from_numpy(label).int(),\
               image_info





