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
from copy import deepcopy
import numpy as np
import torch
import pickle
from monai import data, transforms
from monai.data import *
from monai.transforms import *
from torch.utils.data import DataLoader, ConcatDataset
from utils.data_utils_tumor import *
from utils.data_trans import *
from utils.dataloader_bdmap import *
from utils.voco_trans import VoCoAugmentation

# headneck
headneck_data_root = '/data/'
headneck_cache_root = '/data/cache/headneck/'
others_cache_root = '/data/cache/'

HNSCC_dir = headneck_data_root + "HNSCC_convert_v1/"
HNSCC_jsonlist = "./jsons/HNSCC.json"
HNSCC_list = load_decathlon_datalist(HNSCC_jsonlist, False, "training", base_dir=HNSCC_dir)
HNSCC_cache_dir = headneck_cache_root + 'HNSCC'

QIN_HeadNeck_dir = headneck_data_root + "QIN_convert_v1/"
QIN_HeadNeck_jsonlist = "./jsons/QIN_HeadNeck.json"
QIN_HeadNeck_list = load_decathlon_datalist(QIN_HeadNeck_jsonlist, False, "training", base_dir=QIN_HeadNeck_dir)
QIN_HeadNeck_cache_dir = headneck_cache_root + 'QIN_HeadNeck'

HeadNeckPET_dir = headneck_data_root + "HNPC_convert_v1/"
HeadNeckPET_jsonlist = "./jsons/HeadNeckPet.json"
HeadNeckPET_list = load_decathlon_datalist(HeadNeckPET_jsonlist, False, "training", base_dir=HeadNeckPET_dir)
HeadNeckPET_cache_dir = headneck_cache_root + 'HeadNeckPET'

TCGA_HNSC_dir = headneck_data_root + "TCGA-HNSC_convert_v1/"
TCGA_HNSC_jsonlist = "./jsons/TCGA_HNSC.json"
TCGA_HNSC_list = load_decathlon_datalist(TCGA_HNSC_jsonlist, False, "training", base_dir=TCGA_HNSC_dir)
TCGA_HNSC_cache_dir = headneck_cache_root + 'TCGA_HNSC'

# Totalsegmentator
Totalsegmentator_dir = headneck_data_root + "Totalsegmentator_dataset/"
Totalsegmentator_jsonlist = "./jsons/Totalsegmentator_dataset.json"
Totalsegmentator_list = load_decathlon_datalist(Totalsegmentator_jsonlist, False, "training", base_dir=Totalsegmentator_dir)
Totalsegmentator_cache_dir = others_cache_root + 'Totalsegmentator'

# Colon
CT_COLONOGRAPHY_dir = headneck_data_root + "CT_COLONOGRAPHY_converted_v1/"
CT_COLONOGRAPHY_jsonlist = "./jsons/ColonographyTrials.json"
CT_COLONOGRAPHY_list = load_decathlon_datalist(CT_COLONOGRAPHY_jsonlist, False, "training", base_dir=CT_COLONOGRAPHY_dir)
CT_COLONOGRAPHY_cache_dir = others_cache_root + 'CT_COLONOGRAPHY'


def get_ds_headneck(args):
    if args.distributed:
        print_cond = (args.rank == 0)
    else:
        print_cond = True

    if print_cond:
        print('---' * 20)
        print("Dataset 1 HNSCC: number of data: {}".format(len(HNSCC_list)))
        print("Dataset 2 QIN_HeadNeck: number of data: {}".format(len(QIN_HeadNeck_list)))
        print("Dataset 3 HeadNeckPET: number of data: {}".format(len(HeadNeckPET_list)))
        print("Dataset 4 TCGA_HNSC: number of data: {}".format(len(TCGA_HNSC_list)))

        print("Dataset 5 Totalsegmentator: number of data: {}".format(len(Totalsegmentator_list)))
        print("Dataset 6 CT_COLONOGRAPHY: number of data: {}".format(len(CT_COLONOGRAPHY_list)))

        total_list = HNSCC_list + QIN_HeadNeck_list + HeadNeckPET_list + TCGA_HNSC_list + Totalsegmentator_list + CT_COLONOGRAPHY_list
        print("Total HeadNeck, Totalsegmentator, and Colon: number of data: {}".format(len(total_list)))
        print('---' * 20)

    base_trans = get_headneck_trans(args)

    if args.cache:
        HNSCC_ds = PersistentDataset(data=HNSCC_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=HNSCC_cache_dir)
        QIN_HeadNeck_ds = PersistentDataset(data=QIN_HeadNeck_list,
                                     transform=base_trans,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir=QIN_HeadNeck_cache_dir)

        HeadNeckPET_ds = PersistentDataset(data=HeadNeckPET_list,
                                            transform=base_trans,
                                            pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                            cache_dir=HeadNeckPET_cache_dir)
        TCGA_HNSC_ds = PersistentDataset(data=TCGA_HNSC_list,
                                            transform=base_trans,
                                            pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                            cache_dir=TCGA_HNSC_cache_dir)

        Totalsegmentator_ds = PersistentDataset(data=Totalsegmentator_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=Totalsegmentator_cache_dir)
        CT_COLONOGRAPHY_ds = PersistentDataset(data=CT_COLONOGRAPHY_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=CT_COLONOGRAPHY_cache_dir)
    else:
        HNSCC_ds = Dataset(data=HNSCC_list,transform=transforms.Compose(base_trans))
        QIN_HeadNeck_ds = Dataset(data=QIN_HeadNeck_list, transform=transforms.Compose(base_trans))

        HeadNeckPET_ds = Dataset(data=HeadNeckPET_list, transform=transforms.Compose(base_trans))
        TCGA_HNSC_ds = Dataset(data=TCGA_HNSC_list, transform=transforms.Compose(base_trans))

        Totalsegmentator_ds = Dataset(data=Totalsegmentator_list,transform=transforms.Compose(base_trans))
        CT_COLONOGRAPHY_ds = Dataset(data=CT_COLONOGRAPHY_list,transform=transforms.Compose(base_trans))

    train_ds = ConcatDataset(
        [
        HNSCC_ds, QIN_HeadNeck_ds,
        HeadNeckPET_ds, TCGA_HNSC_ds,
        Totalsegmentator_ds, CT_COLONOGRAPHY_ds
        ])

    return train_ds


def get_loader_headneck(args):

    train_ds = get_ds_headneck(args)
    train_sampler = Sampler(train_ds) if args.distributed else None
    train_loader = data.DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.workers,
        sampler=train_sampler,
        pin_memory=True,
    )

    return train_loader
