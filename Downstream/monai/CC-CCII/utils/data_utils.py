# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os
import pickle
import numpy as np
import torch

from monai import data, transforms
from monai.data import *
import pandas as pd
import random


def get_loader(args):
    '''Get the dataloader for the CCII dataset.'''
    # Transforms
    def __transforms__(augmentation=True, npy=None, args=None):
        RANDOM_BRIGHTNESS = 7
        RANDOM_CONTRAST = 5
        pre_size = 420
        final_size = 384
        spatial_limit = int((pre_size-final_size)/2.0)
        # pre_top_left = int((512-pre_size)/2.0)
        final_top_left = int((512-final_size)/2.0)

        npy_normalized = npy.astype(np.float32) / 255.0 # cast to float
        if augmentation:
            # random flip
            if random.uniform(0, 1) < 0.5: #horizontal flip
                npy_normalized = np.flipud(npy_normalized)
            # color jitter
            br = random.randint(-RANDOM_BRIGHTNESS, RANDOM_BRIGHTNESS) / 100.
            npy_normalized = npy_normalized + br
            # Random contrast
            cr = 1.0 + random.randint(-RANDOM_CONTRAST, RANDOM_CONTRAST) / 100.
            npy_normalized = npy_normalized * cr
            # clip values to 0-1 range
            npy_normalized = np.clip(npy_normalized, 0, 1.0)
            # random crop
            offset_x = random.randint(-spatial_limit, spatial_limit)
            offset_y = random.randint(-spatial_limit, spatial_limit)
            npy_normalized = npy_normalized[
                :,
                final_top_left+offset_x : final_top_left+final_size+offset_x,
                final_top_left+offset_y : final_top_left+final_size+offset_y
                ]
        else:
            npy_normalized = npy_normalized[
                :,
                final_top_left : final_top_left+final_size,
                final_top_left : final_top_left+final_size
                ]
        return npy_normalized

    train_files_name = os.path.join(args.csv_list, f'CC_CCII_fold{args.fold}_train.csv')
    val_files_name = os.path.join(args.csv_list, f'CC_CCII_fold{args.fold}_valid.csv')
    train_files = pd.read_csv(train_files_name)
    val_files = pd.read_csv(val_files_name)

    train_ds = CC_CCII(data=train_files, transforms=__transforms__, augmentation=True, args=args)
    print(f'=>Train len {len(train_ds)}')
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=8, pin_memory=True, persistent_workers=True,
    )

    val_ds = CC_CCII(data=val_files, transforms=__transforms__, augmentation=False,args=args)
    print(f'=>Val len {len(val_ds)}')
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True, persistent_workers=True)
    return train_loader, val_loader


class CC_CCII(torch.utils.data.Dataset):
    '''CC_CCII Covid-19 classification dataset.
    This dataset is used for Covid-19 classification.
    It loads the data from the given directory and csv file.
    The data is preprocessed and augmented using various techniques.
    http://ncov-ai.big.ac.cn/download?lang=en
    '''
    def __init__(self, data=None, transforms=None, augmentation=True, args=None):
        super().__init__()
        self.augmentation = augmentation
        self.df_meta = pd.read_csv(os.path.join(args.csv_list, 'CC_CCII_metadata.csv'))

        df = data
        self.patients = df['patient_id']
        self.scans = df['scan_id']
        self.targets = df['target']
        self.transforms = transforms
        self.args = args

    def __getitem__(self, index):
        target = int(self.targets[index])
        npy = np.load(
            os.path.join(
                self.args.data_dir,
                'p'+str(self.patients[index])+'-s'+str(self.scans[index])+'.npy'
                )
            )

        meta = self.df_meta[(self.df_meta['patient_id'] == self.patients[index])]
        covariates = [
            'Age',
            'Sex(Male1/Female2)',
            'Critical_illness',
            'Liver_function',
            'Lung_function',
            'Progression (Days)'
        ]
        if meta.size == 0:
            meta = np.array([47, 1.5, 0, 1, 2, 6.89],dtype='f8')
        else:
            meta = meta.sample(frac=1.0, replace=True, weights=None, random_state=0, axis=0)
            meta = np.squeeze(meta[covariates].to_numpy(), axis=0)
        meta[0] = np.clip(meta[0] / 100, 0.25, 0.95)
        meta[1] = meta[1] - 1
        meta[3] = meta[3] / 5
        meta[4] = meta[4] / 5
        meta[-1] = meta[-1] / 14

        npy_normalized = self.transforms(self.augmentation, npy, self.args)
        npy_normalized = npy_normalized[np.newaxis,]
        return {
            'image': npy_normalized,
            'label': target
        }

    def __len__(self):
        return len(self.targets)