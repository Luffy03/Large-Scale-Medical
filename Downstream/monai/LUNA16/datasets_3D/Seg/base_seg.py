from torch.utils.data import Dataset
import torchvision.transforms.functional as tf
import torch
from torchvision import transforms
import numpy as np
from scipy import ndimage
import random
import SimpleITK as sitk


class SegmentationBaseTrainset(Dataset):
    """
    Base_train_dataset for segmentation.
    """
    def __init__(self,
                 config,
                 base_dir,
                 flag='train',
                 ):
        super(SegmentationBaseTrainset, self).__init__()
        self.flag = flag
        self.config = config
        self.crop_size = config.input_size
        self.num_classes = config.class_num
        self._base_dir = base_dir
        self.all_images = []

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        pass

    def get_config(self):
       pass

    def _get_img_gt_pair(self, img_path, label_path):

        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        img_array = sitk.GetArrayFromImage(img)

        mask = sitk.ReadImage(label_path, sitk.sitkFloat32)
        mask_array = sitk.GetArrayFromImage(mask)

        # img_array: [D, H, W], mask_array: [D, H, W]

        img_array = np.expand_dims(img_array, 0).astype(np.float32)

        if self.num_classes == 1:
            assert np.max(mask_array) == 1.
            mask_onehot_array = np.expand_dims(mask_array, 0).astype(np.int32)
        else:
            mask_onehot_array = self.create_one_hot_label(mask_array)

        # img_array: [C, D, H, W], mask_onehot_array: [K, D, H, W]

        return img_array, mask_onehot_array

    def create_one_hot_label(self, label):
        """
        Input label: [D, H, W].
        Output label: [K, D, H, W]. The output label contains the background class in the 0th channel.
        """

        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2]))
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.int32)

        return onehot_label

    def random_flip(self, image, label, prob=0.5):
        """Flip an image randomly.
             Args:
                 image: [C, D, H, W]
                 label: [K, D, H, W]
                 axis: [-1, -2], only flip along x and y axis.
        """
        axes = [-1, -2]
        for degree in axes:
            if random.random() < prob:
                image = np.flip(image, axis=degree).copy()
                label = np.flip(label, axis=degree).copy()

    def random_rotation(self, image, label, angle_range):
        """
            Rotate an image randomly.
            Args:
                  image: [C, D, H, W]
                  label: [K, D, H, W], one-hot label

            """
        angle = np.random.randint(angle_range[0], angle_range[1])
        # axes = [(0, 1), (1, 2), (0, 2)]
        axes = [(-2, -1), (-3, -2), (-3, -1)]

        k = np.random.randint(0, 3)
        image = ndimage.interpolation.rotate(image, angle=angle, axes=axes[k], reshape=False, order=1)
        label = label.astype(np.float32)
        label = ndimage.interpolation.rotate(label, angle=angle, axes=axes[k], reshape=False, order=0)
        # label[label >= 0.8] = 1
        # label[label < 0.8] = 0
        image[image < 0] = 0
        image[image > 1] = 1
        label = label.astype(np.int32)
        image = image.astype(np.float32)
        return image, label

    def center_crop(self, image, label, size):
        """CenterCrop an image.
           Args:
              image: [C, D, H, W]
              label:[K, D, H, W]
              crop_size: the desired output size.
            Returns:
              out_image:[C, D, size, size]
              out_label:[K, D, size, size]
        """
        _, d, h, w = image.shape

        w1 = int(round((w - size) / 2.))
        h1 = int(round((h - size) / 2.))

        image = image[:, :, h1:h1 + size, w1:w1 + size]
        label = label[:, :, h1:h1 + size, w1:w1 + size]
        return image, label

    def random_crop(self, image, label):
        """Crop the image in a sample randomly.
              Args:
                  image:[C, D, H, W]
                  label:[[K, D, H, W]
                  crop_size: the desired output size: [d, h, w]
              Returns:
                  out_image:[C, d, h, w]
                  out_label:[K, d, h, w]
               """
        _, d, h, w = image.shape
        # if np.random.uniform() > 0.33:
        #     w1 = np.random.randint((w - self.output_size[0])//4, 3*(w - self.output_size[0])//4)
        #     h1 = np.random.randint((h - self.output_size[1])//4, 3*(h - self.output_size[1])//4)
        # else:
        d1 = np.random.randint(0, d - self.crop_size[0])
        h1 = np.random.randint(0, h - self.crop_size[1])
        w1 = np.random.randint(0, w - self.crop_size[2])

        image = image[:, d1:d1 + self.crop_size[0], h1:h1 + self.crop_size[1], w1:w1 + self.crop_size[2]]
        label = label[:, d1:d1 + self.crop_size[0], h1:h1 + self.crop_size[1], w1:w1 + self.crop_size[2]]

        return image, label

    def random_crop_fg_very_close_to_center(self, image, label, mask_class):
        """Crop the image around foreground in a sample randomly.
            Args:
                  image:[C, D, H, W]
                  label:[K, D, H, W]
                  crop_size: the desired output size [d, h, w]
                  mask_class: use which foreground class to crop.
                                1: organ+tumor 2: only tumor
            Returns:
                  out_image:[C, d, h, w]
                  out_label:[K, d, h, w]
           """

        # crop alongside with the ground truth

        _, d, h, w = image.shape

        if self.num_classes == 1:
            mask = label[0]
            index = np.where(mask == 1)
        else:
            if mask_class == '2':
                mask = label[2]
                if np.max(mask) != 1:
                    # there is none pixel of the tumor.
                    mask = np.zeros_like(label[0])
                    for k in range(1, self.num_classes):
                        mask = np.maximum(mask, label[k] == 1)
                    index = np.where(mask == 1)
                else:
                    index = np.where(mask == 1)

            else:
                # get the union set of classes belonging to the foreground.
                mask = np.zeros_like(label[0])
                for k in range(1, self.num_classes):
                    mask = np.maximum(mask, label[k] == 1)
                index = np.where(mask == 1)

        z_min = np.min(index[0])
        z_max = np.max(index[0])
        x_min = np.min(index[1])
        x_max = np.max(index[1])
        y_min = np.min(index[2])
        y_max = np.max(index[2])

        # middle point
        z_mid = np.int((z_min + z_max) / 2)
        x_mid = np.int((x_min + x_max) / 2)
        y_mid = np.int((y_min + y_max) / 2)
        Delta_z = np.int((z_max - z_min) / 6) + 1  # 4
        Delta_x = np.int((x_max - x_min) / 3) + 1 # 4
        Delta_y = np.int((y_max - y_min) / 3) + 1 # 4

        # random number of x, y, z

        z_random = np.random.randint(z_mid - Delta_z, z_mid + Delta_z)
        x_random = np.random.randint(x_mid - Delta_x, x_mid + Delta_x)
        y_random = np.random.randint(y_mid - Delta_y, y_mid + Delta_y)

        # crop patch
        crop_z_down = z_random - np.int(self.crop_size[0] / 2)
        crop_z_up = z_random + np.int(self.crop_size[0] / 2)
        crop_x_down = x_random - np.int(self.crop_size[1] / 2)
        crop_x_up = x_random + np.int(self.crop_size[1] / 2)
        crop_y_down = y_random - np.int(self.crop_size[2] / 2)
        crop_y_up = y_random + np.int(self.crop_size[2] / 2)

        # If the cube is out of bounds, then use padding.
        if crop_z_down < 0 or crop_z_up > d:
            delta_z = np.maximum(np.abs(crop_z_down), np.abs(crop_z_up - d))
            image = np.pad(image, ((0, 0), (delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=-2.287)
            label = np.pad(label, ((0, 0), (delta_z, delta_z), (0, 0), (0, 0)), 'constant', constant_values=0.0)
            crop_z_down = crop_z_down + delta_z
            crop_z_up = crop_z_up + delta_z

        if crop_x_down < 0 or crop_x_up > h:
            delta_x = np.maximum(np.abs(crop_x_down), np.abs(crop_x_up - h))
            image = np.pad(image, ((0, 0), (0, 0), (delta_x, delta_x), (0, 0)), 'constant', constant_values=-2.287)
            label = np.pad(label, ((0, 0), (0, 0), (delta_x, delta_x), (0, 0)), 'constant', constant_values=0.0)
            crop_x_down = crop_x_down + delta_x
            crop_x_up = crop_x_up + delta_x

        if crop_y_down < 0 or crop_y_up > w:
            delta_y = np.maximum(np.abs(crop_y_down), np.abs(crop_y_up - w))
            image = np.pad(image, ((0, 0), (0, 0), (0, 0), (delta_y, delta_y)), 'constant', constant_values=-2.287)
            label = np.pad(label, ((0, 0), (0, 0), (0, 0), (delta_y, delta_y)), 'constant', constant_values=0.0)
            crop_y_down = crop_y_down + delta_y
            crop_y_up = crop_y_up + delta_y

        label = label[:, crop_z_down: crop_z_up, crop_x_down: crop_x_up, crop_y_down: crop_y_up]
        image = image[:, crop_z_down: crop_z_up, crop_x_down: crop_x_up, crop_y_down: crop_y_up]

        return image, label

    def random_crop_fg(self, image, label):
        """Crop the image around foreground in a sample randomly.
            Args:
                  image:[C, D, H, W]
                  label:[K, D, H, W]
                  crop_size: the desired output size [d, h, w]
            Returns:
                  out_image:[C, d, h, w]
                  out_label:[K, d, h, w]
           """
        _, d, h, w = image.shape

        if self.num_classes == 1:
            mask = label[0]
            index = np.where(mask == 1)
        else:
            # get the union set of classes belonging to the foreground.
            mask = np.zeros_like(label[0])
            for k in range(1, self.num_classes):
                mask = np.maximum(mask, label[k] == 1)
            index = np.where(mask == 1)

        # label [K, D, H, W], index [D, H, W]
        z_min = min(index[0])
        z_max = max(index[0])
        x_min = min(index[1])
        x_max = max(index[1])
        y_min = min(index[2])
        y_max = max(index[2])

        delta = 3

        z_mid = random.randint(z_min + int((z_max - z_min) / delta), z_max - int((z_max - z_min) / delta))
        x_mid = random.randint(x_min + int((x_max - x_min) / delta), x_max - int((x_max - x_min) / delta))
        y_mid = random.randint(y_min + int((y_max - y_min) / delta), y_max - int((y_max - y_min) / delta))

        half_crop_z = int(self.crop_size[0] / 2)
        half_crop_x = int(self.crop_size[1] / 2)
        half_crop_y = int(self.crop_size[2] / 2)

        # no padding
        if x_mid - half_crop_x > 0 and x_mid + half_crop_x < h \
                and y_mid - half_crop_y > 0 and y_mid + half_crop_y < w\
                and z_mid - half_crop_z > 0 and z_mid + half_crop_z < d:
            crop_img = image[:,  z_mid - half_crop_z:z_mid + half_crop_z, x_mid - half_crop_x:x_mid + half_crop_x,
                       y_mid - half_crop_y:y_mid + half_crop_y]
            crop_label = label[:,  z_mid - half_crop_z:z_mid + half_crop_z, x_mid - half_crop_x:x_mid + half_crop_x,
                         y_mid - half_crop_y:y_mid + half_crop_y]

        else:
            # padding
            z_index_min = max(z_mid - half_crop_z, 0)
            z_index_max = min(z_mid + half_crop_z, d)
            x_index_min = max(x_mid - half_crop_x, 0)
            x_index_max = min(x_mid + half_crop_x, h)
            y_index_min = max(y_mid - half_crop_y, 0)
            y_index_max = min(y_mid + half_crop_y, w)

            padding_z_min = max(half_crop_z - z_mid, 0)
            padding_z_max = max(z_mid + half_crop_z - d, 0)
            padding_x_min = max(half_crop_x - x_mid, 0)
            padding_x_max = max(x_mid + half_crop_x - h, 0)
            padding_y_min = max(half_crop_y - y_mid, 0)
            padding_y_max = max(y_mid + half_crop_y - w, 0)

            crop_img = image[:, z_index_min:z_index_max, x_index_min:x_index_max, y_index_min:y_index_max]
            crop_img = np.pad(crop_img, (
                (0, 0), (padding_z_min, padding_z_max), (padding_x_min, padding_x_max), (padding_y_min, padding_y_max)
                ), 'constant', constant_values=0)

            crop_label = label[:, z_index_min:z_index_max, x_index_min:x_index_max, y_index_min:y_index_max]
            crop_label = np.pad(crop_label, (
                (0, 0),  (padding_z_min, padding_z_max), (padding_x_min, padding_x_max), (padding_y_min, padding_y_max)),
                'constant', constant_values=0)
        return crop_img, crop_label

    def __str__(self):
        return 'SegmentationBaseTrainset'


class SegmentationBaseTestset(Dataset):
    """
     Base_test_dataset for segmentation.
    """
    def __init__(self,
                 config,
                 base_dir,
                 flag='train',
                 ):
        super(SegmentationBaseTestset, self).__init__()
        self.flag = flag
        self.config = config
        self.num_classes = config.class_num
        self._base_dir = base_dir
        self.all_images = []
        self.cut_params = config.test_cut_params

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
        img_path, label_path, image_index = self.all_images[index]
        image, label = self._get_img_gt_pair(img_path, label_path)
        print(label)
        org_shape = np.array(image.shape)

        # padding
        full_image = self.padding_image(image, self.cut_params)
        new_shape = np.array(full_image.shape)

        # cut patches
        patches = self.extract_ordered_overlap(full_image, self.cut_params)

        # image info
        image_info = {'org_shape': torch.from_numpy(org_shape), 'new_shape': torch.from_numpy(new_shape), 'image_index':image_index}

        # patches: [N, C, pd, ph, pw], label [K, D, H, W]
        return torch.from_numpy(image.astype(np.float32)),\
               torch.from_numpy(full_image.astype(np.float32)),\
               torch.from_numpy(patches.astype(np.float32)),\
               torch.from_numpy(label.astype(np.int32)).long(),\
               image_info

    def _get_img_gt_pair(self, img_path, target_path):

        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        img_array = sitk.GetArrayFromImage(img)

        mask = sitk.ReadImage(target_path, sitk.sitkFloat32)
        mask_array = sitk.GetArrayFromImage(mask)

        # img_array: [D, H, W], mask_array: [D, H, W]

        img_array = np.expand_dims(img_array, 0).astype(np.float32)

        if self.num_classes == 1:
            assert np.max(mask_array) == 1.
            mask_onehot_array = np.expand_dims(mask_array, 0).astype(np.int32)
        else:
            mask_onehot_array = self.create_one_hot_label(mask_array)

        # img_array: [C, D, H, W], mask_onehot_array: [K, D, H, W]

        return img_array, mask_onehot_array

    def create_one_hot_label(self, label):
        """
        Input label: [D, H, W].
        Output label: [K, D, H, W]. The output label contains the background class in the 0th channel.
        """

        onehot_label = np.zeros(
            (self.num_classes, label.shape[0], label.shape[1], label.shape[2])).astype(np.int32)
        for i in range(self.num_classes):
            onehot_label[i, :, :, :] = (label == i).astype(np.int32)

        return onehot_label

    def padding_image(self, image, cut_params):
        """Padding the test image for subsequent cutting.
        image: [C, D, H, W]
        """
        assert (len(image.shape) == 4)
        c, d, h, w = image.shape
        leftover_d = (d - cut_params['patch_d']) % cut_params['stride_d']
        leftover_h = (h - cut_params['patch_h']) % cut_params['stride_h']
        leftover_w = (w - cut_params['patch_w']) % cut_params['stride_w']

        if (leftover_d != 0):
            new_d = d + (cut_params['stride_d'] - leftover_d)
        else:
            new_d = d

        if (leftover_h != 0):
            new_h = h + (cut_params['stride_h'] - leftover_h)
        else:
            new_h = h

        if (leftover_w != 0):
            new_w = w + (cut_params['stride_w'] - leftover_w)
        else:
            new_w = w

        tmp_full_imgs = np.zeros((c, new_d, new_h, new_w)).astype(np.float32)
        tmp_full_imgs[:, :d, :h, :w] = image
        print("new images shape: \n" + str(image.shape))
        return tmp_full_imgs

    def extract_ordered_overlap(self, image, cut_params):
        """Cut an image according to the test_cut_params.
        image:[C, D, H, W]
        patches: N - [C, pd, ph, pw]
        """
        assert (len(image.shape) == 4)
        c, d, h, w = image.shape
        assert ((h - cut_params['patch_h']) % cut_params['stride_h'] == 0
                and (w - cut_params['patch_w']) % cut_params['stride_w'] == 0
                and (d - cut_params['patch_d']) % cut_params['stride_d'] == 0)

        N_patches_d = (d - cut_params['patch_d']) // cut_params['stride_d'] + 1
        N_patches_h = (h - cut_params['patch_h']) // cut_params['stride_h'] + 1
        N_patches_w = (w - cut_params['patch_w']) // cut_params['stride_w'] + 1

        N_patches = N_patches_d * N_patches_h * N_patches_w
        print("Number of patches d/h/w : ", N_patches_d, N_patches_h, N_patches_w)
        print("number of patches per image: ", N_patches)
        patches = np.empty((N_patches, c, cut_params['patch_d'], cut_params['patch_h'], cut_params['patch_w'])).astype(np.float32)
        iter_tot = 0
        for i in range(N_patches_d):
            for j in range(N_patches_h):
                for k in range(N_patches_w):
                    # if stride_d > patch_d, no overlap.
                    patch = image[:, i * cut_params['stride_d']: i * cut_params['stride_d'] + cut_params['patch_d'],
                                  j * cut_params['stride_h']: j * cut_params['stride_h'] + cut_params['patch_h'],
                                  k * cut_params['stride_w']: k * cut_params['stride_w'] + cut_params['patch_w']]
                    patches[iter_tot] = patch
                    iter_tot += 1
        assert (iter_tot == N_patches)
        return patches

    def extract_ordered_overlap_pair(self, image, label, cut_params):
        """Cut an image according to the test_cut_params.
        image:[C, D, H, W]
        patches: N - [C, pd, ph, pw]
        """
        assert (len(image.shape) == 4)
        c, d, h, w = image.shape
        assert ((h - cut_params['patch_h']) % cut_params['stride_h'] == 0
                and (w - cut_params['patch_w']) % cut_params['stride_w'] == 0
                and (d - cut_params['patch_d']) % cut_params['stride_d'] == 0)

        N_patches_d = (d - cut_params['patch_d']) // cut_params['stride_d'] + 1
        N_patches_h = (h - cut_params['patch_h']) // cut_params['stride_h'] + 1
        N_patches_w = (w - cut_params['patch_w']) // cut_params['stride_w'] + 1

        N_patches = N_patches_d * N_patches_h * N_patches_w
        print("Number of patches d/h/w : ", N_patches_d, N_patches_h, N_patches_w)
        print("number of patches per image: ", N_patches)
        patches = np.empty((N_patches, c, cut_params['patch_d'], cut_params['patch_h'], cut_params['patch_w'])).astype(
            np.float32)
        patch_labels = np.empty((N_patches, label.shape[0], cut_params['patch_d'], cut_params['patch_h'], cut_params['patch_w'])).astype(
            np.float32)
        iter_tot = 0
        for i in range(N_patches_d):
            for j in range(N_patches_h):
                for k in range(N_patches_w):
                    # if stride_d > patch_d, no overlap.
                    patch = image[:, i * cut_params['stride_d']: i * cut_params['stride_d'] + cut_params['patch_d'],
                            j * cut_params['stride_h']: j * cut_params['stride_h'] + cut_params['patch_h'],
                            k * cut_params['stride_w']: k * cut_params['stride_w'] + cut_params['patch_w']]
                    patches[iter_tot] = patch

                    patch_label = label[:, i * cut_params['stride_d']: i * cut_params['stride_d'] + cut_params['patch_d'],
                            j * cut_params['stride_h']: j * cut_params['stride_h'] + cut_params['patch_h'],
                            k * cut_params['stride_w']: k * cut_params['stride_w'] + cut_params['patch_w']]
                    patches[iter_tot] = patch
                    patch_labels[iter_tot] = patch_label
                    iter_tot += 1
        assert (iter_tot == N_patches)
        return patches, patch_labels

    def __str__(self):
        return 'SegmentationBaseTestset'
