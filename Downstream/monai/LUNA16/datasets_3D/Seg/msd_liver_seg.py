import torch
import numpy as np
import random
import os
from .base_seg import SegmentationBaseTrainset, SegmentationBaseTestset
import csv


class SegmentationMSDLIVERTrainSet(SegmentationBaseTrainset):
    """
    Training dataset for segmentation in the MSD dataset (MSD).
    MSD: segment both the liver and tumors in each image.
    """
    def __init__(self,
                 config,
                 base_dir,
                 flag='train',
                 ):
        super(SegmentationMSDLIVERTrainSet, self).__init__(config, base_dir, flag)
        self.flag = flag
        self.config = config
        self.train_patch_fg_ratio = config.train_patch_fg_ratio
        self.crop_size = config.input_size
        self.num_classes = config.class_num
        self._base_dir = base_dir
        self.order = config.order

        # load data
        with open(os.path.join(self._base_dir, flag + '.csv')) as f:
            reader = csv.reader(f)
            for row in reader:
                if self.config.train_dataset.find('down2') != -1:
                    row = row[0].replace('stage1', 'stage0')
                    self.all_images.append(row)
                else:
                    self.all_images.append(row[0])

        print(self.all_images)
        assert len(self.all_images) != 0, "the images can`t be zero!"
        ### Display status
        print('Number of images in {}: {:d}'.format(flag, len(self.all_images)))

    def __len__(self):
        return len(self.all_images)

    def _get_img_gt_pair_from_npz(self, img_path):

        data_array = np.load(img_path, allow_pickle=True)
        img_array = data_array['data'][0]
        mask_array = data_array['data'][1]

        # img_array: [D, H, W], mask_array: [D, H, W]
        img_array = np.expand_dims(img_array, 0).astype(np.float32)

        if self.num_classes == 1:
            assert np.max(mask_array) == 1.
            mask_onehot_array = np.expand_dims(mask_array, 0).astype(np.int32)
        else:
            mask_onehot_array = self.create_one_hot_label(mask_array)

        # img_array: [C, D, H, W], mask_onehot_array: [K, D, H, W]

        return img_array, mask_onehot_array

    def complement(self, image_org, label_org):
        c, d, h, w = image_org.shape
        sub_d = self.crop_size[0] - d
        sub_h = self.crop_size[1] - h
        completed_img = []
        completed_label = []

        if sub_d >= 0:
            completed_img = np.zeros((c, self.crop_size[0]+1, h, w)).astype(np.float32)
            completed_label = np.zeros((self.num_classes, self.crop_size[0]+1, h, w)).astype(np.float32)
            completed_img[:, :d, :, :] = image_org
            completed_label[:, :d, :, :] = label_org

        if sub_h >= 0:
            completed_img = np.zeros((c, d, self.crop_size[1]+1, self.crop_size[2]+1)).astype(np.float32)
            completed_label = np.zeros((self.num_classes, d, self.crop_size[1]+1, self.crop_size[2]+1)).astype(np.float32)
            completed_img[:, :, :h, :w] = image_org
            completed_label[:, :, :h, :w] = label_org

        return completed_img, completed_label

    def __getitem__(self, index):
         # only for 3D patch seg along x, y, z axes
        image_path = self.all_images[index]
        image_index = os.path.split(image_path)[1][:-4]

        image_org, label_org = self._get_img_gt_pair_from_npz(image_path)

        if self.config.train_dataset.find('down2') != -1:
            if image_org.shape[1] <= self.crop_size[0] or image_org.shape[2] <= self.crop_size[1]:
                # Complete the volumes with size less than desired crop size.
                image_org, label_org = self.complement(image_org, label_org)

        # random crop
        prob = random.random()

        if prob < self.train_patch_fg_ratio * (2 / 3):
            image, label = self.random_crop_fg(image_org, label_org)
        elif prob < self.train_patch_fg_ratio:
            image, label = self.random_crop_fg_very_close_to_center(image_org, label_org, mask_class='2')
        else:
            image, label = self.random_crop(image_org, label_org)

        # if prob < self.train_patch_fg_ratio * (1 / 2):
        #      image, label = self.random_crop_fg(image_org, label_org)
        # elif prob < self.train_patch_fg_ratio:
        #     image, label = self.random_crop_fg_very_close_to_center(image_org, label_org, mask_class='2')
        # else:
        #     image, label = self.random_crop(image_org, label_org)

        return torch.from_numpy(image).float(), \
               torch.from_numpy(label).float(), \
               image_index

    def __str__(self):
       pass


class SegmentationMSDLIVERTestset(SegmentationBaseTestset):
    """
    Test dataset for segmentation in the MSD dataset (MSD).
    MSD: segment both the liver and tumors in each image.
    """
    def __init__(self, config, base_dir, flag='train'):
        super(SegmentationMSDLIVERTestset, self).__init__(config, base_dir, flag)
        self.config = config
        self._base_dir = base_dir
        self.all_images = []
        self.order = config.order
        # load data
        with open(os.path.join(self._base_dir, flag + '.csv')) as f:
            reader = csv.reader(f)
            for row in reader:
                if self.config.eval_dataset.find('down2') != -1:
                    row = row[0].replace('stage1', 'stage0')
                    self.all_images.append(row)
                else:
                    self.all_images.append(row[0])

        assert len(self.all_images) != 0, "the images can`t be zero!"

        ### Display status
        print('Number of images in {}: {:d}'.format(flag, len(self.all_images)))

    def __len__(self):
        return len(self.all_images)

    def _get_img_gt_pair_from_npz(self, img_path):

        data_array = np.load(img_path, allow_pickle=True)
        img_array = data_array['data'][0]
        mask_array = data_array['data'][1]

        # img_array: [D, H, W], mask_array: [D, H, W]
        img_array = np.expand_dims(img_array, 0).astype(np.float32)

        if self.num_classes == 1:
            assert np.max(mask_array) == 1.
            mask_onehot_array = np.expand_dims(mask_array, 0).astype(np.int32)
        else:
            mask_onehot_array = self.create_one_hot_label(mask_array)

        # img_array: [C, D, H, W], mask_onehot_array: [K, D, H, W]

        return img_array, mask_onehot_array

    def __getitem__(self, index):
        image_path = self.all_images[index]
        image_index = os.path.split(image_path)[1][:-4]

        image, label = self._get_img_gt_pair_from_npz(image_path)

        org_shape = np.array(label.shape)

        # padding
        full_image = self.padding_image(image, self.cut_params)
        # full_label = self.padding_image(label, self.cut_params)
        new_shape = np.array(full_image.shape)

        # cut patches
        patches = self.extract_ordered_overlap(full_image, self.cut_params)
        # patches, patch_labels = self.extract_ordered_overlap_pair(full_image, full_label, self.cut_params)

        # image info
        image_info = {'org_shape': torch.from_numpy(org_shape),
                      'new_shape': torch.from_numpy(new_shape),
                      'image_index': image_index}

        # patches: [N, C, pd, ph, pw], label [K, D, H, W]
        return torch.from_numpy(image),\
               torch.from_numpy(full_image),\
               torch.from_numpy(patches),\
               torch.from_numpy(label).int(), \
               image_info







