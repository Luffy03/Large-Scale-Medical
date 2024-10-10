import json
import multiprocessing
import os
from pathlib import Path
from typing import List, Tuple, Union

import pandas as pd
import SimpleITK as sitk
from tqdm import tqdm

# from resampling import resample_image_to_spacing
# from totalsegmentator_class_map import CLASS_MAP_ALL



import SimpleITK as sitk


SITK_INTERPOLATOR_DICT = {
    'nearest': sitk.sitkNearestNeighbor,
    'linear': sitk.sitkLinear,
    'gaussian': sitk.sitkGaussian,
    'label_gaussian': sitk.sitkLabelGaussian,
    'bspline': sitk.sitkBSpline,
    'hamming_sinc': sitk.sitkHammingWindowedSinc,
    'cosine_windowed_sinc': sitk.sitkCosineWindowedSinc,
    'welch_windowed_sinc': sitk.sitkWelchWindowedSinc,
    'lanczos_windowed_sinc': sitk.sitkLanczosWindowedSinc
}


def resample_image_to_spacing(image, new_spacing, default_value, interpolator='linear'):
    assert interpolator in SITK_INTERPOLATOR_DICT, \
        (f"Interpolator '{interpolator}' not part of SimpleITK. "
         f"Please choose one of the following {list(SITK_INTERPOLATOR_DICT.keys())}.")
    assert image.GetDimension() == len(new_spacing), \
        (f"Input is {image.GetDimension()}-dimensional while "
         f"the new spacing is {len(new_spacing)}-dimensional.")

    interpolator = SITK_INTERPOLATOR_DICT[interpolator]
    spacing = image.GetSpacing()
    size = image.GetSize()
    new_size = [int(round(siz * spac / n_spac)) for siz, spac, n_spac in zip(size, spacing, new_spacing)]
    return sitk.Resample(
        image,
        new_size,             # size
        sitk.Transform(),     # transform
        interpolator,         # interpolator
        image.GetOrigin(),    # outputOrigin
        new_spacing,          # outputSpacing
        image.GetDirection(), # outputDirection
        default_value,        # defaultPixelValue
        image.GetPixelID()    # outputPixelType
    )


"""
TotalSegmentator has a full resolution approach and a faster, lower resolution approach.
The full resolution approach is trained on isotropic 1.5mm CT scans, while the lower resolution
approach is trained on 3mm CT scans.

The lower resolution approach uses a single nnUNet model to predict all 104 classes.
The full resolution approach uses 5 nnUNet models, each predicting 21 classes (total of 104).
"""

CLASS_MAP_ALL = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "aorta",
    8: "inferior_vena_cava",
    9: "portal_vein_and_splenic_vein",
    10: "pancreas",
    11: "adrenal_gland_right",
    12: "adrenal_gland_left",
    13: "lung_upper_lobe_left",
    14: "lung_lower_lobe_left",
    15: "lung_upper_lobe_right",
    16: "lung_middle_lobe_right",
    17: "lung_lower_lobe_right",
    18: "vertebrae_L5",
    19: "vertebrae_L4",
    20: "vertebrae_L3",
    21: "vertebrae_L2",
    22: "vertebrae_L1",
    23: "vertebrae_T12",
    24: "vertebrae_T11",
    25: "vertebrae_T10",
    26: "vertebrae_T9",
    27: "vertebrae_T8",
    28: "vertebrae_T7",
    29: "vertebrae_T6",
    30: "vertebrae_T5",
    31: "vertebrae_T4",
    32: "vertebrae_T3",
    33: "vertebrae_T2",
    34: "vertebrae_T1",
    35: "vertebrae_C7",
    36: "vertebrae_C6",
    37: "vertebrae_C5",
    38: "vertebrae_C4",
    39: "vertebrae_C3",
    40: "vertebrae_C2",
    41: "vertebrae_C1",
    42: "esophagus",
    43: "trachea",
    44: "heart_myocardium",
    45: "heart_atrium_left",
    46: "heart_ventricle_left",
    47: "heart_atrium_right",
    48: "heart_ventricle_right",
    49: "pulmonary_artery",
    50: "brain",
    51: "iliac_artery_left",
    52: "iliac_artery_right",
    53: "iliac_vena_left",
    54: "iliac_vena_right",
    55: "small_bowel",
    56: "duodenum",
    57: "colon",
    58: "rib_left_1",
    59: "rib_left_2",
    60: "rib_left_3",
    61: "rib_left_4",
    62: "rib_left_5",
    63: "rib_left_6",
    64: "rib_left_7",
    65: "rib_left_8",
    66: "rib_left_9",
    67: "rib_left_10",
    68: "rib_left_11",
    69: "rib_left_12",
    70: "rib_right_1",
    71: "rib_right_2",
    72: "rib_right_3",
    73: "rib_right_4",
    74: "rib_right_5",
    75: "rib_right_6",
    76: "rib_right_7",
    77: "rib_right_8",
    78: "rib_right_9",
    79: "rib_right_10",
    80: "rib_right_11",
    81: "rib_right_12",
    82: "humerus_left",
    83: "humerus_right",
    84: "scapula_left",
    85: "scapula_right",
    86: "clavicula_left",
    87: "clavicula_right",
    88: "femur_left",
    89: "femur_right",
    90: "hip_left",
    91: "hip_right",
    92: "sacrum",
    93: "face",
    94: "gluteus_maximus_left",
    95: "gluteus_maximus_right",
    96: "gluteus_medius_left",
    97: "gluteus_medius_right",
    98: "gluteus_minimus_left",
    99: "gluteus_minimus_right",
    100: "autochthon_left",
    101: "autochthon_right",
    102: "iliopsoas_left",
    103: "iliopsoas_right",
    104: "urinary_bladder"
}

CLASS_MAP_5_PARTS = {
    # 17 classes
    "organs": {
        1: "spleen",
        2: "kidney_right",
        3: "kidney_left",
        4: "gallbladder",
        5: "liver",
        6: "stomach",
        7: "aorta",
        8: "inferior_vena_cava",
        9: "portal_vein_and_splenic_vein",
        10: "pancreas",
        11: "adrenal_gland_right",
        12: "adrenal_gland_left",
        13: "lung_upper_lobe_left",
        14: "lung_lower_lobe_left",
        15: "lung_upper_lobe_right",
        16: "lung_middle_lobe_right",
        17: "lung_lower_lobe_right"
    },

    # 24 classes
    "vertebrae": {
        1: "vertebrae_L5",
        2: "vertebrae_L4",
        3: "vertebrae_L3",
        4: "vertebrae_L2",
        5: "vertebrae_L1",
        6: "vertebrae_T12",
        7: "vertebrae_T11",
        8: "vertebrae_T10",
        9: "vertebrae_T9",
        10: "vertebrae_T8",
        11: "vertebrae_T7",
        12: "vertebrae_T6",
        13: "vertebrae_T5",
        14: "vertebrae_T4",
        15: "vertebrae_T3",
        16: "vertebrae_T2",
        17: "vertebrae_T1",
        18: "vertebrae_C7",
        19: "vertebrae_C6",
        20: "vertebrae_C5",
        21: "vertebrae_C4",
        22: "vertebrae_C3",
        23: "vertebrae_C2",
        24: "vertebrae_C1"
    },

    # 18
    "cardiac": {
        1: "esophagus",
        2: "trachea",
        3: "heart_myocardium",
        4: "heart_atrium_left",
        5: "heart_ventricle_left",
        6: "heart_atrium_right",
        7: "heart_ventricle_right",
        8: "pulmonary_artery",
        9: "brain",
        10: "iliac_artery_left",
        11: "iliac_artery_right",
        12: "iliac_vena_left",
        13: "iliac_vena_right",
        14: "small_bowel",
        15: "duodenum",
        16: "colon",
        17: "urinary_bladder",
        18: "face"
    },

    # 21
    "muscles": {
        1: "humerus_left",
        2: "humerus_right",
        3: "scapula_left",
        4: "scapula_right",
        5: "clavicula_left",
        6: "clavicula_right",
        7: "femur_left",
        8: "femur_right",
        9: "hip_left",
        10: "hip_right",
        11: "sacrum",
        12: "gluteus_maximus_left",
        13: "gluteus_maximus_right",
        14: "gluteus_medius_left",
        15: "gluteus_medius_right",
        16: "gluteus_minimus_left",
        17: "gluteus_minimus_right",
        18: "autochthon_left",
        19: "autochthon_right",
        20: "iliopsoas_left",
        21: "iliopsoas_right"
    },

    # 24 classes
    # 12. ribs start from vertebrae T12
    # Small subset of population (roughly 8%) have 13. rib below 12. rib
    #  (would start from L1 then)
    #  -> this has label rib_12
    # Even smaller subset (roughly 1%) has extra rib above 1. rib   ("Halsrippe")
    #  (the extra rib would start from C7)
    #  -> this has label rib_1
    #
    # Quite often only 11 ribs (12. ribs probably so small that not found). Those
    # cases often wrongly segmented.
    "ribs": {
        1: "rib_left_1",
        2: "rib_left_2",
        3: "rib_left_3",
        4: "rib_left_4",
        5: "rib_left_5",
        6: "rib_left_6",
        7: "rib_left_7",
        8: "rib_left_8",
        9: "rib_left_9",
        10: "rib_left_10",
        11: "rib_left_11",
        12: "rib_left_12",
        13: "rib_right_1",
        14: "rib_right_2",
        15: "rib_right_3",
        16: "rib_right_4",
        17: "rib_right_5",
        18: "rib_right_6",
        19: "rib_right_7",
        20: "rib_right_8",
        21: "rib_right_9",
        22: "rib_right_10",
        23: "rib_right_11",
        24: "rib_right_12"
    }
}


def merge_masks(segmentations_path: str, class_map: dict) -> sitk.Image:
    """Merge the masks of a patient into a single mask. The masks are merged according to the class map.
    When masks overlap, the latter mask will overwrite the former mask in those areas.

    Args:
        segmentations_path (str): Path to the segmentations directory of a patient.
        class_map (dict): A dictionary mapping the class names to their label values.

    Returns:
        sitk.Image: The merged mask.
    """

    # Read all masks and assign them their label value based on `class_map`
    for i, (label_value, label) in enumerate(class_map.items()):
        mask_path = Path(segmentations_path) / f"{label}.nii.gz"

        try:
            print(f'normal mask')
            mask = sitk.ReadImage(str(mask_path), sitk.sitkUInt8)
        except:
            print(f'cosines problem occures on mask, try to fix it...')
            import nibabel as nib
            img = nib.load(str(mask_path))
            qform = img.get_qform()
            img.set_qform(qform)
            sform = img.get_sform()
            img.set_sform(sform)

            check_dir('./temp/mask' + str(Path(segmentations_path)))
            nib.save(img, './temp/mask' + str(mask_path))

            mask = sitk.ReadImage('./temp/mask' + str(mask_path), sitk.sitkUInt8)
            print(f'now we have fixed mask!')

        mask = mask * label_value

        # The first mask is the base mask, all other masks are added to it
        if i == 0:
            combined_mask = mask
            continue

        # Add the mask to the base mask.
        # When masks overlap, the latter mask will overwrite the former mask in those areas.
        # https://github.com/wasserth/TotalSegmentator/issues/8#issuecomment-1222364214
        try:
            combined_mask = sitk.Maximum(combined_mask, mask)
        except RuntimeError:
            print(f"Failed to add mask {label} for {segmentations_path.parent.name},"
                  " likely due to different physical space. Enforcing the same space and retrying.")
            mask.SetSpacing(combined_mask.GetSpacing())
            mask.SetDirection(combined_mask.GetDirection())
            mask.SetOrigin(combined_mask.GetOrigin())
            combined_mask = sitk.Maximum(combined_mask, mask)
            print("Success!")

    return combined_mask


def check_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def process_patient(patient: str,
                    output_dir: str,
                    target_spacing: Union[List, Tuple],
                    class_map: dict,
                    df: pd.DataFrame) -> None:
    """Resample the images and masks to the target spacing, merge the masks, and save them
    to the target directory in the nnUNet format.

    Args:
        patient (str): path to the patient directory.
        output_dir (str): path to the output directory.
        target_spacing (Union[List, Tuple]): spacing to resample the images and masks to.
        class_map (dict): A dictionary mapping the class names to their label values.
        df (pd.DataFrame): The metadata provided in the TotalSegmentator dataset loaded as a DataFrame.
    """

    # Resample the images and masks to the target spacing, merge the masks, and save them to the target directory
    try:
        print(f'normal')
        scan = sitk.ReadImage(str(patient / "ct.nii.gz"))
    except:
        print(f'cosines problem occures, try to fix it...')
        import nibabel as nib
        img = nib.load(str(patient / "ct.nii.gz"))
        qform = img.get_qform()
        img.set_qform(qform)
        sform = img.get_sform()
        img.set_sform(sform)

        check_dir('./temp/'+str(patient))
        nib.save(img, './temp/'+str(patient / "ct.nii.gz"))

        scan = sitk.ReadImage('./temp/'+str(patient / "ct.nii.gz"))
        print(f'now we have fixed it!')

    # scan = sitk.ReadImage(str(patient / "ct.nii.gz"))

    scan = resample_image_to_spacing(scan, new_spacing=target_spacing, default_value=-1024, interpolator="linear")

    # Merge the masks according to the class map
    mask = merge_masks(patient / "segmentations", class_map)
    mask = resample_image_to_spacing(mask, new_spacing=target_spacing, default_value=0, interpolator="nearest")
    mask = sitk.Cast(mask, sitk.sitkUInt8)

    # Get the split (train, val, test) of the patient
    split = df.loc[df["image_id"] == patient.name, "split"].values[0]
    # nnUNet's naming is "imagesTr" for train and "imagesTs" for test, there is no val split directory.
    # Instead, it used cross-validation. However, TotalSegmentator has a predefined train, val, and test split,
    # and we achieve that by copying the `splits_final.json` into the nnUNet's preprocessed directory of the dataset.
    train_or_test = "Ts" if split == "test" else "Tr"

    # TotalSegmentator's naming is "sXXXX" (e.g., s0191). Get the last 4 characters to use as the nnUNet case identifier.
    case_identifier = patient.name[-4:]
    scan_output_path = output_dir / f"images{train_or_test}/TotalSegmentator_{case_identifier}_0000.nii.gz"
    mask_output_path = output_dir / f"labels{train_or_test}/TotalSegmentator_{case_identifier}.nii.gz"

    sitk.WriteImage(scan, str(scan_output_path), useCompression=True)
    sitk.WriteImage(mask, str(mask_output_path), useCompression=True)


def create_dataset(input_dir: str,
                   output_dir: str,
                   target_spacing: list,
                   class_map: dict,
                   num_cores: int = -1) -> None:
    """Create the dataset for nnUNet v2 by resampling the images and masks to the target spacing,
    merging the masks, and saving them to the target directory in the nnUNet format.
    Support multiprocessing by using the `num_cores` argument.

    Args:
        input_dir (str): TotalSegmentator dataset directory.
        output_dir (str): nnUNet v2 raw dataset directory.
        target_spacing (list): spacing to resample the images and masks to.
        class_map (dict): A dictionary mapping the class names to their label values.
        num_cores (int, optional): Number of cores to use. Defaults to -1, which means all cores.
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Get all patient directories
    patients = [x for x in input_dir.iterdir() if x.is_dir()]

    # Read the metadata provided in the TotalSegmentator dataset
    df = pd.read_csv(input_dir / "meta.csv", delimiter=";")

    # Create the dataset.json file required by nnUNet
    dataset_json = {
        "channel_names": {"0": "CT"},
        # nnUNet v2 requries the the label names to be keys, and the label values to be values, flip them.
        "labels": {v: k for k, v in class_map.items()} | {"background": 0},
        # Equal to the train and val splits combined as nnUNet does cross-validation by default.
        "numTraining": df.loc[df["split"] != "test"].shape[0],
        "file_ending": ".nii.gz"
    }

    # Save the dataset.json file
    with open(output_dir / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4, sort_keys=True)

    # Create the imagesTr, imagesTs, labelsTr, labelsTs directories
    for name in ["imagesTr", "imagesTs", "labelsTr", "labelsTs"]:
        (output_dir / name).mkdir(exist_ok=True, parents=True)

    # Multiprocessing
    if num_cores == -1:
        print("All cores selected.")
        num_cores = os.cpu_count()
    if num_cores > 1:
        print(f"Running in multiprocessing mode with cores: {num_cores}.")
        with multiprocessing.Pool(processes=num_cores) as pool:
            pool.starmap(process_patient, [(patient, output_dir, target_spacing, class_map, df) for patient in patients])
    else:
        print("Running in main process only.")
        for patient in tqdm(patients):
            process_patient(patient, output_dir, target_spacing, class_map, df)


if __name__ == "__main__":
    create_dataset(
        input_dir="/data/jiaxin/data/Totalsegmentator_dataset/",
        output_dir="/data/linshan/Dataset606_Totalsegmentator/",
        target_spacing=[1.5, 1.5, 1.5],
        class_map=CLASS_MAP_ALL,
        num_cores=16
    )