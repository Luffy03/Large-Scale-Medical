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


data_dir = "/data/"
cache_dir = '/data/cache/'

BTCV_dir = data_dir + "BTCV"
BTCV_jsonlist = os.path.join(BTCV_dir, "dataset_0.json")
BTCV_trainlist = load_decathlon_datalist(BTCV_jsonlist, True, "training", base_dir=BTCV_dir)
BTCV_vallist = load_decathlon_datalist(BTCV_jsonlist, True, "validation", base_dir=BTCV_dir)
BTCV_list = BTCV_trainlist + BTCV_vallist
BTCV_cache_dir = cache_dir + 'BTCV'

flare_dir = data_dir + "Flare22/"
flare_json = os.path.join(flare_dir, "dataset.json")
flare_trainlist = load_decathlon_datalist(flare_json, True, "training", base_dir=flare_dir)
flare_vallist = load_decathlon_datalist(flare_json, True, "validation", base_dir=flare_dir)
flare_list = flare_trainlist + flare_vallist
flare_cache_dir = cache_dir + 'flare22'

Amos_dir = data_dir + "Amos2022/"
Amos_json = os.path.join(Amos_dir, "dataset_CT.json")
Amos_trainlist = load_decathlon_datalist(Amos_json, True, "training", base_dir=Amos_dir)
Amos_vallist = load_decathlon_datalist(Amos_json, True, "validation", base_dir=Amos_dir)
Amos_list = Amos_trainlist + Amos_vallist
Amos_cache_dir = cache_dir + 'amos'

Word_dir = data_dir + "WORD/"
Word_json = os.path.join(Word_dir, "dataset.json")
word_train_list = load_decathlon_datalist(Word_json, True, "training", base_dir=Word_dir)
word_val_list = load_decathlon_datalist(Word_json, True, "validation", base_dir=Word_dir)
word_train_list = word_train_list + word_val_list
word_test_list = load_decathlon_datalist(Word_json, True, "test", base_dir=Word_dir)
Word_list = word_train_list + word_test_list
Word_cache_dir = cache_dir + 'word'

flare23_dir = data_dir + 'Flare23/'
flare23_json = "./jsons/flare23_new.json"
flare23_list = load_decathlon_datalist(flare23_json, True, "training", base_dir=flare23_dir)
flare23_cache_dir = cache_dir + 'flare23'

DeepLesion_dir = data_dir + 'DeepLesion/'
DeepLesion_json = "./jsons/DeepLesion.json"
DeepLesion_list = load_decathlon_datalist(DeepLesion_json, True, "training", base_dir=DeepLesion_dir)
DeepLesion_cache_dir = cache_dir + 'DeepLesion'

PANORAMA_dir = data_dir + 'PANORAMA/'
PANORAMA_json = "./jsons/PANORAMA.json"
PANORAMA_list = load_decathlon_datalist(PANORAMA_json, True, "training", base_dir=PANORAMA_dir)
PANORAMA_cache_dir = cache_dir + 'PANORAMA'


def get_loader_abdomen(args):
    train_ds = get_ds_abdomen(args)

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


def get_ds_abdomen(args):
    if args.distributed:
        print_cond = (args.rank == 0)
    else:
        print_cond = True

    if print_cond:
        print('---' * 20)
        print("Dataset 1 BTCV: number of data: {}".format(len(BTCV_list)))
        print("Dataset 2 Flare22: number of data: {}".format(len(flare_list)))
        print("Dataset 3 Amos: number of data: {}".format(len(Amos_list)))
        print("Dataset 4 Word: number of data: {}".format(len(Word_list)))
        print("Dataset 5 Flare23: number of data: {}".format(len(flare23_list)))
        print("Dataset 6 Tumor: number of data: {}".format(len(lits_list)+len(panc_list)+len(kits_list)+len(healthy_ct_list)))
        print("Dataset 7 AbdomenAtlas: number of data: {}".format(5195))
        print("Dataset 8 DeepLesion: number of data: {}".format(len(DeepLesion_list)))
        print("Dataset 9 PANORAMA: number of data: {}".format(len(PANORAMA_list)))

        total_list = BTCV_list + flare_list + Amos_list + Word_list + flare23_list + lits_list + panc_list + kits_list + healthy_ct_list + DeepLesion_list + PANORAMA_list
        print("Total Abdomen: number of data: {}".format(len(total_list)+5195))
        print('---' * 20)

    base_trans = get_abdomen_trans(args)
    base_trans_without_label = get_abdomen_trans_without_label(args)

    if args.cache:
        BTCV_ds = PersistentDataset(data=BTCV_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=BTCV_cache_dir)
        flare_ds = PersistentDataset(data=flare_list,
                                     transform=base_trans,
                                     pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir=flare_cache_dir)
        Amos_ds = PersistentDataset(data=Amos_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=Amos_cache_dir)
        WORD_ds = PersistentDataset(data=Word_list,
                                    transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=Word_cache_dir)
        flare23_ds = PersistentDataset(data=flare23_list,
                                       transform=base_trans,
                                       pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                       cache_dir=flare23_cache_dir)

        PANORAMA_ds = PersistentDataset(data=PANORAMA_list,
                                       transform=base_trans,
                                       pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                       cache_dir=PANORAMA_cache_dir)

        tumor_ds = get_loader_tumor(args)

        # AbdomenAtlas 5k
        Atlas_train_ds = get_atlas_ds(args)

        # New dataset without labels
        DeepLesion_ds = PersistentDataset(data=DeepLesion_list,
                                       transform=base_trans_without_label,
                                       pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                       cache_dir=DeepLesion_cache_dir)
    else:
        BTCV_ds = Dataset(data=BTCV_list,transform=transforms.Compose(base_trans))
        flare_ds = Dataset(data=flare_list, transform=transforms.Compose(base_trans))
        Amos_ds = Dataset(data=Amos_list,transform=transforms.Compose(base_trans))
        WORD_ds = Dataset(data=Word_list,transform=transforms.Compose(base_trans))
        flare23_ds = Dataset(data=flare23_list, transform=transforms.Compose(base_trans))

        PANORAMA_ds = Dataset(data=PANORAMA_list,transform=transforms.Compose(base_trans))

        tumor_ds = get_loader_tumor(args)

        # AbdomenAtlas 5k
        Atlas_train_ds = get_atlas_ds(args)

        # New dataset without labels
        DeepLesion_ds = Dataset(data=DeepLesion_list, transform=transforms.Compose(base_trans_without_label))

    train_ds = ConcatDataset(
        [
         tumor_ds,
         BTCV_ds, flare_ds, Amos_ds, WORD_ds,
         flare23_ds,
         Atlas_train_ds,
         DeepLesion_ds,
         PANORAMA_ds
        ])

    return train_ds
