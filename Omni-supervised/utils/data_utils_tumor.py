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
from monai.losses import *
import pickle
from utils.data_trans import *
from monai.config import KeysCollection

lits_dir = '/data/Dataset003_Liver/'
lits_json = "./jsons/dataset_lits.json"
lits_train_list = load_decathlon_datalist(lits_json, True, "training", base_dir=lits_dir)
lits_val_list = load_decathlon_datalist(lits_json, True, "validation", base_dir=lits_dir)
lits_list = lits_train_list + lits_train_list
lits_cache_dir = '/data/cache/LiTs'

panc_dir = '/data/Dataset007_Pancreas/'
panc_json = './jsons/dataset_panc.json'
panc_train_list = load_decathlon_datalist(panc_json, True, "training", base_dir=panc_dir)
panc_val_list = load_decathlon_datalist(panc_json, True, "validation", base_dir=panc_dir)
panc_list = panc_train_list + panc_val_list
panc_cache_dir = '/data/cache/Pancreas'

kits_dir = '/data/Dataset220_KiTS2023/'
kits_json = './jsons/dataset_kits.json'
kits_train_list = load_decathlon_datalist(kits_json, True, "training", base_dir=kits_dir)
kits_val_list = load_decathlon_datalist(kits_json, True, "validation", base_dir=kits_dir)
kits_list = kits_train_list + kits_val_list
kits_cache_dir = '/data/cache/KiTs'

healthyCT_dir = '/data/healthy_ct/'
healthy_ct_liver_json = "./jsons/healthy_ct_liver.json"
healthy_ct_panc_json = "./jsons/healthy_ct_panc.json"
healthy_ct_kidney_json = "./jsons/healthy_ct_kidney.json"

healthy_ct_liver_list = load_decathlon_datalist(healthy_ct_liver_json, True, "training", base_dir=healthyCT_dir)
healthy_ct_panc_list = load_decathlon_datalist(healthy_ct_liver_json, True, "training", base_dir=healthyCT_dir)
healthy_ct_kidney_list = load_decathlon_datalist(healthy_ct_liver_json, True, "training", base_dir=healthyCT_dir)

healthy_ct_list = healthy_ct_liver_list + healthy_ct_panc_list + healthy_ct_kidney_list
healthy_ct_cache_dir = '/data/cache/healthCT'


def get_loader_tumor(args):
    base_trans = get_abdomen_trans(args)

    if args.cache:
        lits_ds = PersistentDataset(data=lits_list, transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=lits_cache_dir)
        panc_ds = PersistentDataset(data=panc_list, transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=panc_cache_dir)
        kits_ds = PersistentDataset(data=kits_list, transform=base_trans,
                                    pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                    cache_dir=kits_cache_dir)

        health_ds = PersistentDataset(data=healthy_ct_list, transform=base_trans, pickle_protocol=pickle.HIGHEST_PROTOCOL,
                                     cache_dir=healthy_ct_cache_dir)
    else:
        lits_ds = Dataset(data=lits_list, transform=transforms.Compose(base_trans))
        panc_ds = Dataset(data=panc_list, transform=transforms.Compose(base_trans))
        kits_ds = Dataset(data=kits_list, transform=transforms.Compose(base_trans))

        health_ds = Dataset(data=healthy_ct_list, transform=transforms.Compose(base_trans))

    train_ds = ConcatDataset(
        [health_ds,
         lits_ds, panc_ds, kits_ds])

    return train_ds