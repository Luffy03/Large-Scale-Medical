import copy
import random
import time

import numpy as np
import torch
from PIL import Image
from scipy.special import comb
from torch.utils.data import Dataset
import torchio.transforms


class RKBBase(Dataset):
    def __init__(self, config, base_dir, flag='train'):
        self.config = config
        self.base_dir = base_dir
        self.all_images = []
        self.flag = flag
        self.crop_size = config.input_size
        self.org_data_size = config.org_data_size
        self.gaps = config.gaps
        self.num_grids_per_axis = config.num_grids_per_axis
        self.num_cubes = self.num_grids_per_axis ** 3

        self.order_num_classes = config.order_class_num
        self.rot_num_classes = self.num_cubes

        self.K_permutations = np.load(config.k_permutations_path)
        assert self.order_num_classes == len(self.K_permutations)

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        pass

    def crop_3d(self, image, flag, crop_size):
        h, w, d = crop_size[0], crop_size[1], crop_size[2]
        h_old, w_old, d_old = image.shape[0], image.shape[1], image.shape[2]

        if flag == 'train':
            # crop random
            x = np.random.randint(0, 1 + h_old - h)
            y = np.random.randint(0, 1 + w_old - w)
            z = np.random.randint(0, 1 + d_old - d)
        else:
            # crop center
            x = int((h_old - h) / 2)
            y = int((w_old - w) / 2)
            z = int((d_old - d) / 2)

        return self.do_crop_3d(image, x, y, z, h, w, d)

    def do_crop_3d(self, image, x, y, z, h, w, d):
        assert type(x) == int, x
        assert type(y) == int, y
        assert type(z) == int, z
        assert type(h) == int, h
        assert type(w) == int, w
        assert type(d) == int, d

        return image[x:x + h, y:y + w, z:z + d]

    def crop_cubes_3d(self, image, flag, cubes_per_side, cube_jitter_xy=3, cube_jitter_z=3):
        h, w, d = image.shape

        patch_overlap = -cube_jitter_xy if cube_jitter_xy < 0 else 0

        h_grid = (h - patch_overlap) // cubes_per_side
        w_grid = (w - patch_overlap) // cubes_per_side
        d_grid = (d - patch_overlap) // cubes_per_side
        h_patch = h_grid - cube_jitter_xy
        w_patch = w_grid - cube_jitter_xy
        d_patch = d_grid - cube_jitter_z

        cubes = []
        for i in range(cubes_per_side):
            for j in range(cubes_per_side):
                for k in range(cubes_per_side):

                    p = self.do_crop_3d(image,
                                   i * h_grid,
                                   j * w_grid,
                                   k * d_grid,
                                   h_grid + patch_overlap,
                                   w_grid + patch_overlap,
                                   d_grid + patch_overlap)

                    if h_patch < h_grid or w_patch < w_grid or d_patch < d_grid:
                        p = self.crop_3d(p, flag, [h_patch, w_patch, d_patch])

                    cubes.append(p)

        return cubes

    def rearrange(self, cubes, K_permutations):
        label = random.randint(0, len(K_permutations) - 1)
        # print('label', np.array(K_permutations[label]), label)
        return np.array(cubes)[np.array(K_permutations[label])], label

    def center_crop_xy(self, image, size):
        """CenterCrop a sample.
           Args:
              image: [D, H, W]
              label:[D, H, W]
              crop_size: the desired output size in the x-y plane
            Returns:
              out_image:[D, h, w]
              out_label:[D, h, w]
        """
        h, w, d = image.shape

        h1 = int(round((h - size[0]) / 2.))
        w1 = int(round((w - size[1]) / 2.))

        image = image[h1:h1 + size[0], w1:w1 + size[1], :]
        return image

    def rotate(self, cubes):

        # multi-hot labels
        # [8, H, W, D]
        rot_cubes = copy.deepcopy(cubes)
        hor_vector = []
        ver_vector = []

        for i in range(self.num_cubes):
            p = random.random()
            cube = rot_cubes[i]
            # [H, W, D]
            if p < 1/3:
                hor_vector.append(1)
                ver_vector.append(0)
                # rotate 180 along x axis
                rot_cubes[i] = np.flip(cube, (1, 2))
            elif p < 2/3:
                hor_vector.append(0)
                ver_vector.append(1)
                # rotate 180 along z axis
                rot_cubes[i] = np.flip(cube, (0, 1))

            else:
                hor_vector.append(0)
                ver_vector.append(0)

        return rot_cubes, hor_vector, ver_vector

    def mask(self, cubes):
        mask_vector = []
        masked_cubes = copy.deepcopy(cubes)
        for i in range(self.num_cubes):
            cube = masked_cubes[i]
            if random.random() < 0.5:
                # mask
                mask_vector.append(1)
                R = np.random.uniform(0, 1, cube.shape)
                R = (R > 0.5).astype(np.int32)
                masked_cubes[i] = cube * R
            else:
                mask_vector.append(0)

        return masked_cubes, mask_vector