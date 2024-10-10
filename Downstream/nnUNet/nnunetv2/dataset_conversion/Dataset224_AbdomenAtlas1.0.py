from batchgenerators.utilities.file_and_folder_operations import *
import shutil
# from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

from typing import Tuple

from batchgenerators.utilities.file_and_folder_operations import save_json, join


def generate_dataset_json(output_folder: str,
                          channel_names: dict,
                          labels: dict,
                          num_training_cases: int,
                          file_ending: str,
                          regions_class_order: Tuple[int, ...] = None,
                          dataset_name: str = None, reference: str = None, release: str = None, license: str = None,
                          description: str = None,
                          overwrite_image_reader_writer: str = None, **kwargs):
    """
    Generates a dataset.json file in the output folder

    channel_names:
        Channel names must map the index to the name of the channel, example:
        {
            0: 'T1',
            1: 'CT'
        }
        Note that the channel names may influence the normalization scheme!! Learn more in the documentation.

    labels:
        This will tell nnU-Net what labels to expect. Important: This will also determine whether you use region-based training or not.
        Example regular labels:
        {
            'background': 0,
            'left atrium': 1,
            'some other label': 2
        }
        Example region-based training:
        {
            'background': 0,
            'whole tumor': (1, 2, 3),
            'tumor core': (2, 3),
            'enhancing tumor': 3
        }

        Remember that nnU-Net expects consecutive values for labels! nnU-Net also expects 0 to be background!

    num_training_cases: is used to double check all cases are there!

    file_ending: needed for finding the files correctly. IMPORTANT! File endings must match between images and
    segmentations!

    dataset_name, reference, release, license, description: self-explanatory and not used by nnU-Net. Just for
    completeness and as a reminder that these would be great!

    overwrite_image_reader_writer: If you need a special IO class for your dataset you can derive it from
    BaseReaderWriter, place it into nnunet.imageio and reference it here by name

    kwargs: whatever you put here will be placed in the dataset.json as well

    """
    has_regions: bool = any([isinstance(i, (tuple, list)) and len(i) > 1 for i in labels.values()])
    if has_regions:
        assert regions_class_order is not None, f"You have defined regions but regions_class_order is not set. " \
                                                f"You need that."
    # channel names need strings as keys
    keys = list(channel_names.keys())
    for k in keys:
        if not isinstance(k, str):
            channel_names[str(k)] = channel_names[k]
            del channel_names[k]

    # labels need ints as values
    for l in labels.keys():
        value = labels[l]
        if isinstance(value, (tuple, list)):
            value = tuple([int(i) for i in value])
            labels[l] = value
        else:
            labels[l] = int(labels[l])

    dataset_json = {
        'channel_names': channel_names,  # previously this was called 'modality'. I didn't like this so this is
        # channel_names now. Live with it.
        'labels': labels,
        'numTraining': num_training_cases,
        'file_ending': file_ending,
    }

    if dataset_name is not None:
        dataset_json['name'] = dataset_name
    if reference is not None:
        dataset_json['reference'] = reference
    if release is not None:
        dataset_json['release'] = release
    if license is not None:
        dataset_json['licence'] = license
    if description is not None:
        dataset_json['description'] = description
    if overwrite_image_reader_writer is not None:
        dataset_json['overwrite_image_reader_writer'] = overwrite_image_reader_writer
    if regions_class_order is not None:
        dataset_json['regions_class_order'] = regions_class_order

    dataset_json.update(kwargs)

    save_json(dataset_json, join(output_folder, 'dataset.json'), sort_keys=False)


if __name__ == '__main__':
    """
    How to train our submission to the JHU benchmark

    1. Execute this script here to convert the dataset into nnU-Net format. Adapt the paths to your system!
    2. Run planning and preprocessing: `nnUNetv2_plan_and_preprocess -d 224 -npfp 64 -np 64 -c 3d_fullres -pl 
    nnUNetPlannerResEncL_torchres`. Adapt the number of processes to your System (-np; -npfp)! Note that each process 
    will again spawn 4 threads for resampling. This custom planner replaces the nnU-Net default resampling scheme with 
    a torch-based implementation which is faster but less accurate. This is needed to satisfy the inference speed 
    constraints.
    3. Run training with `nnUNetv2_train 224 3d_fullres all -p nnUNetResEncUNetLPlans_torchres`. 24GB VRAM required, 
    training will take ~28-30h.
    """

    base = '/project/medimgfmod/CT/AbdomenAtlas1.0Mini'
    cases = subdirs(base, join=False, prefix='BDMAP')

    target_dataset_id = 224
    target_dataset_name = f'Dataset{target_dataset_id:3.0f}_AbdomenAtlas1.0'

    raw_dir = '/project/medimgfmod/CT/nnunet_data/nnUNet_raw/'
    maybe_mkdir_p(join(raw_dir, target_dataset_name))
    imagesTr = join(raw_dir, target_dataset_name, 'imagesTr')
    labelsTr = join(raw_dir, target_dataset_name, 'labelsTr')
    maybe_mkdir_p(imagesTr)
    maybe_mkdir_p(labelsTr)

    for case in cases:
        shutil.copy(join(base, case, 'ct.nii.gz'), join(imagesTr, case + '_0000.nii.gz'))
        shutil.copy(join(base, case, 'combined_labels.nii.gz'), join(labelsTr, case + '.nii.gz'))

    labels = {
        "background": 0,
        "aorta": 1,
        "gall_bladder": 2,
        "kidney_left": 3,
        "kidney_right": 4,
        "liver": 5,
        "pancreas": 6,
        "postcava": 7,
        "spleen": 8,
        "stomach": 9
    }

    generate_dataset_json(
        join(raw_dir, target_dataset_name),
        {0: 'nonCT'},  # this was a mistake we did at the beginning and we keep it like that here for consistency
        labels,
        len(cases),
        '.nii.gz',
        None,
        target_dataset_name,
        overwrite_image_reader_writer='NibabelIOWithReorient'
    )