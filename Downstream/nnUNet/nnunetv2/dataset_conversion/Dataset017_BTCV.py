#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


import multiprocessing
import shutil
from multiprocessing import Pool
from collections import OrderedDict
import SimpleITK as sitk
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.paths import nnUNet_raw


if __name__ == "__main__":
    base = "/data/linshan/CTs/BTCV/"

    task_id = 17
    task_name = "BTCV"
    prefix = 'BTCV'

    foldername = "Dataset%03.0d_%s" % (task_id, task_name)

    out_base = join(nnUNet_raw, foldername)
    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    maybe_mkdir_p(imagestr)
    maybe_mkdir_p(imagests)
    maybe_mkdir_p(labelstr)

    train_folder = join(base, "imagesTr")
    label_folder = join(base, "labelsTr")
    test_folder = join(base, "imagesTs")
    train_patient_names = []
    test_patient_names = []
    train_patients = subfiles(train_folder, join=False, suffix = 'nii.gz')
    for p in train_patients:
        serial_number = int(p[3:7])
        train_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
        label_file = join(label_folder, f'label{p[3:]}')
        image_file = join(train_folder, p)
        shutil.copy(image_file, join(imagestr, f'{train_patient_name[:8]}_0000.nii.gz'))
        shutil.copy(label_file, join(labelstr, train_patient_name))
        train_patient_names.append(train_patient_name)

    test_patients = subfiles(test_folder, join=False, suffix=".nii.gz")
    for p in test_patients:
        p = p[:-7]
        image_file = join(test_folder, p + ".nii.gz")
        serial_number = int(p[3:7])
        test_patient_name = f'{prefix}_{serial_number:03d}.nii.gz'
        shutil.copy(image_file, join(imagests, f'{test_patient_name[:8]}_0000.nii.gz'))
        test_patient_names.append(test_patient_name)

    generate_dataset_json(out_base,
                          channel_names={0: 'CT'},
                          labels={
                                "background":0,
                                "spleen":1,
                                "right kidney":2,
                                "left kidney":3,
                                "gallbladder":4,
                                "esophagus":5,
                                "liver":6,
                                "stomach":7,
                                "aorta":8,
                                "inferior vena cava":9,
                                 "portal vein and splenic vein":10,
                                 "pancreas":11,
                                 "right adrenal gland":12,
                                 "left adrenal gland":13
                          },
                          num_training_cases=len(train_patient_names),
                          file_ending='.nii.gz',
                          license='see challenge website',
                          reference='see https://www.synapse.org/#!Synapse:syn3193805/wiki/217789',
                          dataset_release='0.0')


    # json_dict = OrderedDict()
    # json_dict['name'] = "AbdominalOrganSegmentation"
    # json_dict['description'] = "Multi-Atlas Labeling Beyond the Cranial Vault Abdominal Organ Segmentation"
    # json_dict['tensorImageSize'] = "3D"
    # json_dict['reference'] = "https://www.synapse.org/#!Synapse:syn3193805/wiki/217789"
    # json_dict['licence'] = "see challenge website"
    # json_dict['release'] = "0.0"
    # json_dict['modality'] = {
    #     "0": "CT",
    # }
    # json_dict['labels'] = OrderedDict({
    #     "00": "background",
    #     "01": "spleen",
    #     "02": "right kidney",
    #     "03": "left kidney",
    #     "04": "gallbladder",
    #     "05": "esophagus",
    #     "06": "liver",
    #     "07": "stomach",
    #     "08": "aorta",
    #     "09": "inferior vena cava",
    #     "10": "portal vein and splenic vein",
    #     "11": "pancreas",
    #     "12": "right adrenal gland",
    #     "13": "left adrenal gland"}
    # )
    # json_dict['numTraining'] = len(train_patient_names)
    # json_dict['numTest'] = len(test_patient_names)
    # json_dict['training'] = [{'image': "./imagesTr/%s" % train_patient_name, "label": "./labelsTr/%s" % train_patient_name} for i, train_patient_name in enumerate(train_patient_names)]
    # json_dict['test'] = ["./imagesTs/%s" % test_patient_name for test_patient_name in test_patient_names]
    #
    # save_json(json_dict, os.path.join(out_base, "dataset.json"))