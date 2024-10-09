import torch
import numpy as np
import random
import os
import csv
from datasets_3D.paths import Path
from datasets_3D.Classification.base_classification import ClassificationBase
import argparse
from torch.utils.data import DataLoader


class ClassificationLUNASet(ClassificationBase):
    """
    NCC dataset with data mean-resampling.
    """
    def __init__(self,
                 config,
                 base_dir,
                 flag='train',
                 ):
        super(ClassificationLUNASet, self).__init__(config, base_dir, flag)

        if self.flag == 'train':
            # only for sampling class-balanced data in the training stage.
            self.random_sample_ratio = self.config.random_sample_ratio
        else:
            # for test/validation
            self.random_sample_ratio = None

        if self.random_sample_ratio is not None:
            self.all_images = []
            self.all_labels = []
            # 0: False positive class; 1: True positive class
            self.class_0_images = []
            self.class_0_labels = []
            self.class_1_images = []
            self.class_1_labels = []
            with open(os.path.join(self._base_dir, 'train_0.csv')) as f:
                reader = csv.reader(f)
                for row in reader:
                    self.class_0_labels.append(row[0])
                    self.class_0_images.append(row[1])
            f.close()

            with open(os.path.join(self._base_dir, 'train_1.csv')) as f:
                reader = csv.reader(f)
                for row in reader:
                    self.class_1_labels.append(int(row[0]))
                    self.class_1_images.append(row[1])
            f.close()
            # train0.csv: 274637
            # train1.csv: 7931
            self.random_sampler()
        else:
            # wo random_sampler
            self.all_images = []
            self.all_labels = []
            with open(os.path.join(self._base_dir, flag + '.csv')) as f:
                reader = csv.reader(f)
                for row in reader:
                    self.all_labels.append(int(row[0]))
                    self.all_images.append(row[1])

            f.close()
            # train.csv: 282568
            # valid.csv: 109482
            # test.csv: 166225

        assert len(self.all_images) != 0, "the images can`t be zero!"

        ### Display status
        print('Number of images in {}: {:d}'.format(flag, len(self.all_images)))

    def random_sampler(self):
        print('******Random sampler*****')
        assert self.flag == 'train'
        random.shuffle(self.class_0_images)
        self.all_images = self.class_0_images[:int(len(self.class_1_images)/self.random_sample_ratio)] + self.class_1_images
        self.all_labels = self.class_0_labels[:int(len(self.class_1_images)/self.random_sample_ratio)] + self.class_1_labels
        print('Number of class 1 images in {}: {:d}'.format(self.flag, len(self.class_1_images)))
        print('Number of class 0 images in {}: {:d}'.format(self.flag, len(self.all_images)-len(self.class_1_images)))

    def __getitem__(self, index):
        image_path = self.all_images[index]
        image_array = np.load(image_path).astype(np.float32)
        image_name = os.path.split(image_path)[1][:-4]

        if np.max(image_array) > 1:
            image_array = np.multiply(image_array, 1.0 / 255.0)
            image_array = np.clip(image_array, 0, 1)

        assert np.min(image_array) >= 0 and np.max(image_array) <= 1
        label = np.array(self.all_labels[index]).astype(np.int32)

        # [z, y, x] -> [x, y, z]
        image_array = image_array.transpose((2, 1, 0))
        image_array = np.expand_dims(image_array, 0)
        label = np.expand_dims(label, 0)

        return torch.from_numpy(image_array).float(), torch.from_numpy(label).float(), image_name


if __name__=="__main__":

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.class_num = 2
    dataset = ClassificationLUNASet(args, base_dir=Path.db_root_dir('luna_ncc'), flag="test")
    dataloader = DataLoader(dataset, batch_size=1, num_workers=0, shuffle=False)

    count = 0
    for i, sample in enumerate(dataloader):
        image, label, image_name = sample
        print(label, image_name)
        if i == 15:
            image1 = image[0]
            print(image_name[0])
        # save_path = '../results/Luna_ncc' + str(i)
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        # utils.save_np2nii(image[0][0], save_path, 'img')
