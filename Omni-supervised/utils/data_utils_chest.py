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

# chest
chest_data_root = '/data/'
chest_cache_root = '/data/cache/chest/'

LUNA16_dir = chest_data_root + "Luna16-jx/"
LUNA16_jsonlist = "./jsons/dataset_LUNA16_0.json"
LUNA16_list = load_decathlon_datalist(LUNA16_jsonlist, False, "training", base_dir=LUNA16_dir)
LUNA16_cache_dir = chest_cache_root + 'LUNA16'

TCIAcovid19_dir = chest_data_root + "TCIAcovid19/"
TCIAcovid19_jsonlist = "./jsons/dataset_TCIAcovid19_0.json"
TCIAcovid19_list = load_decathlon_datalist(TCIAcovid19_jsonlist, False, "training", base_dir=TCIAcovid19_dir)
TCIAcovid19_cache_dir = chest_cache_root + 'TCIAcovid19'

stoic21_dir = chest_data_root + "stoic21/"
stoic21_jsonlist = "./jsons/stoic21.json"
stoic21_list = load_decathlon_datalist(stoic21_jsonlist, False, "training", base_dir=stoic21_dir)
stoic21_cache_dir = chest_cache_root + 'stoic21'

LIDC_dir = chest_data_root + "LIDC_convert_v1/"
LIDC_jsonlist = "./jsons/LIDC.json"
LIDC_list = load_decathlon_datalist(LIDC_jsonlist, False, "training", base_dir=LIDC_dir)
LIDC_cache_dir = chest_cache_root + 'LIDC'

StonyBrookChestCT_dir = chest_data_root + "StonyBrookChestCT_v1/"
StonyBrookChestCT_jsonlist = "./jsons/StonyBrookChestCT.json"
StonyBrookChestCT_list = load_decathlon_datalist(StonyBrookChestCT_jsonlist, False, "training", base_dir=StonyBrookChestCT_dir)
StonyBrookChestCT_cache_dir = chest_cache_root + 'StonyBrookChestCT'

MELA_dir = chest_data_root + 'MELA/'
MELA_json = "./jsons/MELA.json"
MELA_list = load_decathlon_datalist(MELA_json, True, "training", base_dir=MELA_dir)
MELA_cache_dir = chest_data_root + 'MELA/'


CTRATE_dir = chest_data_root + "CT-RATE/dataset/train/"
CTRATE_jsonlist = "./jsons/ct_rate.json"
CTRATE_list = load_decathlon_datalist(CTRATE_jsonlist, False, "training", base_dir=CTRATE_dir)
CTRATE_cache_dir = chest_cache_root + 'CTRATE'

NLST_dir = chest_data_root + "NLST_convert_v1/"
NLST_jsonlist = "./jsons/NLST_convert_v1.json"
NLST_list = load_decathlon_datalist(NLST_jsonlist, False, "training", base_dir=NLST_dir)
NLST_cache_dir = chest_cache_root + 'NLST'


def get_ds_chest(args):

    if args.distributed:
        print_cond = (args.rank == 0)
    else:
        print_cond = True

    if print_cond:
        print('---' * 20)
        print("Dataset 1 LUNA16: number of data: {}".format(len(LUNA16_list)))
        print("Dataset 2 TCIAcovid19: number of data: {}".format(len(TCIAcovid19_list)))
        print("Dataset 3 stoic21: number of data: {}".format(len(stoic21_list)))
        print("Dataset 4 LIDC: number of data: {}".format(len(LIDC_list)))
        print("Dataset 5 StonyBrookChestCT: number of data: {}".format(len(StonyBrookChestCT_list)))
        print("Dataset 6 MELA: number of data: {}".format(len(MELA_list)))
        print("Dataset 7 CT-RATE: number of data: {}".format(len(CTRATE_list)))
        print("Dataset 8 NLST: number of data: {}".format(len(NLST_list)))

        total_list = LUNA16_list + TCIAcovid19_list+ stoic21_list + LIDC_list + StonyBrookChestCT_list + MELA_list + CTRATE_list + NLST_list
        print("Total Chest: number of data: {}".format(len(total_list)))
        print('---' * 20)

    base_trans = get_chest_trans(args)

    if args.cache:
        LUNA16_ds = PersistentDataset(data=LUNA16_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=LUNA16_cache_dir)

        TCIAcovid19_ds = PersistentDataset(data=TCIAcovid19_list,
                                     transform=base_trans,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir=TCIAcovid19_cache_dir)
        stoic21_ds = PersistentDataset(data=stoic21_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=stoic21_cache_dir)
        LIDC_ds = PersistentDataset(data=LIDC_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=LIDC_cache_dir)
        StonyBrookChestCT_ds = PersistentDataset(data=StonyBrookChestCT_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=StonyBrookChestCT_cache_dir)

        MELA_ds = PersistentDataset(data=MELA_list,
                                      transform=base_trans,
                                      pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                      cache_dir=MELA_cache_dir)

        CTRATE_ds = PersistentDataset(data=CTRATE_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=CTRATE_cache_dir)

        NLST_ds = PersistentDataset(data=NLST_list,
                                       transform=base_trans,
                                       pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                       cache_dir=NLST_cache_dir)
    else:
        LUNA16_ds = Dataset(data=LUNA16_list,transform=transforms.Compose(base_trans))
        TCIAcovid19_ds = Dataset(data=TCIAcovid19_list, transform=transforms.Compose(base_trans))
        stoic21_ds = Dataset(data=stoic21_list,transform=transforms.Compose(base_trans))
        LIDC_ds = Dataset(data=LIDC_list,transform=transforms.Compose(base_trans))
        StonyBrookChestCT_ds = Dataset(data=StonyBrookChestCT_list, transform=transforms.Compose(base_trans))
        MELA_ds = Dataset(data=MELA_list, transform=transforms.Compose(base_trans))
        CTRATE_ds = Dataset(data=CTRATE_list, transform=transforms.Compose(base_trans))
        NLST_ds = Dataset(data=NLST_list, transform=transforms.Compose(base_trans))

    train_ds = ConcatDataset(
        [
         LUNA16_ds,
         TCIAcovid19_ds, stoic21_ds, LIDC_ds, StonyBrookChestCT_ds,
            MELA_ds,
        CTRATE_ds,
         NLST_ds
        ])

    return train_ds


def get_loader_chest(args):

    train_ds = get_ds_chest(args)

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
