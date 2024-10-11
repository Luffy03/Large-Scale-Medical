import os
from typing import Tuple
import numpy as np
from batchgenerators.utilities.file_and_folder_operations import *


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-7] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, modalities: dict,
                          labels: dict, dataset_name: str, sort_keys=True, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset_CT.json you intend to write, so
    output_file='DATASET_PATH/dataset_CT.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param sort_keys: In order to sort or not, the keys in dataset_CT.json
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in modalities.keys()}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i} for i
        in
        train_identifiers]

    save_json(json_dict, os.path.join(output_file), sort_keys=sort_keys)


if __name__=='__main__':

    generate_dataset_json(output_file='./dataset_unlabeled.json',
                          imagesTr_dir='/data/imagesUn',
                          modalities={"0": "CT"},
                          labels={"0": "background",
                                  "1": "spleen",
                                    "2": "rkid",
                                    "3": "lkid",
                                    "4": "gall",
                                    "5": "eso",
                                    "6": "liver",
                                    "7": "sto",
                                    "8": "aorta",
                                    "9": "IVC",
                                    "10": "veins",
                                    "11": "pancreas",
                                    "12": "rad",
                                    "13": "lad",
                                    "14": "liver tumor",
                                    "15": "panc tumor",
                                    "16": "kidney tumor",
                                    "17": "covid",
                                    "18": "colon",
                                    "19": "colon cancer",
                                    "20": "lung cancer"
                                },

                          dataset_name="semi-unlabeled",
                          dataset_description='https://github.com/Luffy03/Large-Scale-Medical',
                          dataset_reference='[CVPR 2024] VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis')
