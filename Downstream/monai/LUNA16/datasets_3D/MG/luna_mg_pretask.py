import copy
import random
import numpy as np
import torch
from tqdm import tqdm
import os

from .base_mg_pretask import MGBase


class MGLunaPretaskSet(MGBase):
    """
    Luna Dataset for Model Genesis.
     Adapted from https://github.com/MrGiovanni/ModelsGenesis
    """
    def __init__(self, config, base_dir, flag):
        super(MGLunaPretaskSet, self).__init__(config, base_dir, flag)
        self.config = config
        self.flag = flag
        self.crop_size = config.input_size
        if self.flag == 'train':
            self.folds = config.train_fold
        else:
            self.folds = config.valid_fold

        # image deformation
        self.nonlinear_rate = config.nonlinear_rate
        self.paint_rate = config.paint_rate
        self.outpaint_rate = config.outpaint_rate
        self.inpaint_rate = 1.0 - self.outpaint_rate
        self.local_rate = config.local_rate
        self.flip_rate = config.flip_rate
        # load data from .npy
        self.get_luna_list()

        assert len(self.all_images) != 0, "the images can`t be zero!"

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):

        input = self.all_images[index]
        # input [c, h, w, d]

        # Autoencoder
        gt = copy.deepcopy(input)

        # Flipping
        input, gt = self.data_augmentation(input, gt, self.flip_rate)

        # Local Shuffling Pixel
        input = self.local_pixel_shuffling(input, prob=self.local_rate)

        # Apply non-Linear transformation with an assigned probability
        input = self.nonlinear_transformation(input, self.nonlinear_rate)

        # Inpainting & Outpainting
        if random.random() < self.paint_rate:
            if random.random() < self.inpaint_rate:
                # Inpainting
                input = self.image_in_painting(input)
            else:
                # Outpainting
                input = self.image_out_painting(input)

        # return torch.tensor(input.copy(), dtype=torch.float32), torch.tensor(gt.copy(), dtype=torch.float32)
        return torch.from_numpy(input.copy()).float(), torch.from_numpy(gt.copy()).float()

    def get_luna_list(self):
        self.all_images = []
        for i, fold in enumerate(tqdm(self.folds)):
            file_name = "bat_" + str(self.config.scale) + "_s_" + str(self.crop_size[0]) + "x" + str(
                self.crop_size[1]) + "x" + str(
                self.crop_size[2]) + "_" + str(fold) + ".npy"
            print('***file_name**:', file_name)
            s = np.load(os.path.join(self.base_dir, file_name))
            self.all_images.extend(s)
        self.all_images = np.expand_dims(np.array(self.all_images), axis=1)

        print("x_{}: {} | {:.2f} ~ {:.2f}".format(self.flag, self.all_images.shape, np.min(self.all_images), np.max(self.all_images)))
        # x_train: (14159, 1, 64, 64, 32) | 0.00 ~ 1.00
        # x_valid: (5640, 1, 64, 64, 32) | 0.00 ~ 1.00
        return



