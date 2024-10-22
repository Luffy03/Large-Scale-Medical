import os
import glob
import json
import torch
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from functools import partial
import torch.nn.functional as F
import tqdm
import osqp
import nibabel as nib
from monai.transforms import *

class CTReportDatasetinfer(Dataset):
    def __init__(self, data_folder, csv_file, min_slices=20, resize_dim=500, force_num_frames=True, labels = "labels.csv"):
        self.data_folder = data_folder
        self.min_slices = min_slices
        self.labels = labels
        self.accession_to_text = self.load_accession_text(csv_file)
        self.paths=[]
        self.samples = self.prepare_samples()
        self.transform = transforms.Compose([
            transforms.Resize((resize_dim,resize_dim)),
            transforms.ToTensor()
        ])
        self.nii_to_tensor = partial(self.nii_img_to_tensor, transform = self.transform)

    def load_accession_text(self, csv_file):
        df = pd.read_csv(csv_file)
        accession_to_text = {}
        for index, row in df.iterrows():
            accession_to_text[row['VolumeName']] = row["Findings_EN"],row['Impressions_EN']
        return accession_to_text


    def prepare_samples_ori(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))
        # print(patient_folders)
##        with open('/home/csexuefeng/CT-CLIP/scripts/patient_folders_train.json', 'w') as f:
##        with open('/home/csexuefeng/CT-CLIP/scripts/patient_folders_valid.json', 'w') as f:
##            json.dump(patient_folders, f)
##        with open('/home/csexuefeng/CT-CLIP/scripts/patient_folders_train.json', 'r') as file:
##        with open('/home/csexuefeng/CT-CLIP/scripts/patient_folders_valid.json', 'r') as file:
##            patient_folders = json.load(file)
        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        for patient_folder in tqdm.tqdm(patient_folders):
##            print(patient_folder)
            accession_folders = glob.glob(os.path.join(patient_folder, '*'))

            for accession_folder in accession_folders:
                nii_files = glob.glob(os.path.join(accession_folder, '*.nii.gz.npz'))

                for nii_file in nii_files:
                    accession_number = nii_file.split("/")[-1]

##                    accession_number = accession_number.replace(".npz", ".nii.gz")
                    accession_number = accession_number.replace(".npz", "")
                    if accession_number not in self.accession_to_text:
                        continue

                    impression_text = self.accession_to_text[accession_number]
                    text_final = ""
                    for text in list(impression_text):
                        text = str(text)
                        if text == "Not given.":
                            text = ""

                        text_final = text_final + text

                    onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
                    if len(onehotlabels) > 0:
                        samples.append((nii_file, text_final, onehotlabels[0]))
                        self.paths.append(nii_file)
        return samples


    def prepare_samples(self):
        samples = []
        patient_folders = glob.glob(os.path.join(self.data_folder, '*'))

        with open('./ct_rate_preprocess_val.json', 'r') as file:
            nii_files = json.load(file)
        # Read labels once outside the loop
        test_df = pd.read_csv(self.labels)
        test_label_cols = list(test_df.columns[1:])
        test_df['one_hot_labels'] = list(test_df[test_label_cols].values)

        for nii_file in nii_files:
            nii_file = nii_file['image_path']
            accession_number = nii_file.split("/")[-1]

##            accession_number = accession_number.replace(".npz", ".nii.gz")
            accession_number = accession_number.replace(".npz", "")
            if accession_number not in self.accession_to_text:
                continue

            impression_text = self.accession_to_text[accession_number]
            text_final = ""
            for text in list(impression_text):
                text = str(text)
                if text == "Not given.":
                    text = ""

                text_final = text_final + text

            onehotlabels = test_df[test_df["VolumeName"] == accession_number]["one_hot_labels"].values
            if len(onehotlabels) > 0:
                samples.append((nii_file, text_final, onehotlabels[0]))
                self.paths.append(nii_file)
        return samples
    '''
    def prepare_samples(self):
        samples = []
        import json

        ann_path = '/project/medimgfmod/CTRG/label/chest/chest_ctrg_r2gen_nitfy_ori_split.json'
        ann = json.loads(open(ann_path, 'r').read())
        example = ann['val']
        for i in range(len(example)):
            nii_file = '/scratch/medimgfmod/CT/CTRG/Chest_New_nitfy_2/' + example[i]['id']
            input_text_concat = example[i]['report']
            samples.append((nii_file, input_text_concat))
            self.paths.append(nii_file)
        return samples
    '''
    def __len__(self):
        return len(self.samples)

    def nii_img_to_tensor(self, path, transform):
        # vocab
        img_data = np.load(path)['arr_0']
        img_data= np.transpose(img_data, (1, 2, 0))
        img_data = img_data*1000
        hu_min, hu_max = -1000, 200
        img_data = np.clip(img_data, hu_min, hu_max)

        img_data = (((img_data+400 ) / 600)).astype(np.float32)
        slices=[]

        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)
##        target_shape = (256,256,128)
##        target_shape = (96,96,96)
##        target_shape = (90,90,90)
        '''
        # ct clip pretrain
        nii_img = nib.load(str(path))
        img_data = nii_img.get_fdata()
        img_data = img_data.astype(np.float32) / 255.0
        slices=[]
        tensor = torch.tensor(img_data)
        # Get the dimensions of the input tensor
        target_shape = (480,480,240)
        '''
        # Extract dimensions
        h, w, d = tensor.shape

        # Calculate cropping/padding values for height, width, and depth
        dh, dw, dd = target_shape
        h_start = max((h - dh) // 2, 0)
        h_end = min(h_start + dh, h)
        w_start = max((w - dw) // 2, 0)
        w_end = min(w_start + dw, w)
        d_start = max((d - dd) // 2, 0)
        d_end = min(d_start + dd, d)

        # Crop or pad the tensor
        tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

        pad_h_before = (dh - tensor.size(0)) // 2
        pad_h_after = dh - tensor.size(0) - pad_h_before

        pad_w_before = (dw - tensor.size(1)) // 2
        pad_w_after = dw - tensor.size(1) - pad_w_before

        pad_d_before = (dd - tensor.size(2)) // 2
        pad_d_after = dd - tensor.size(2) - pad_d_before

        tensor = torch.nn.functional.pad(tensor, (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after), value=-1)


        tensor = tensor.permute(2, 0, 1)

        tensor = tensor.unsqueeze(0)

        train_transforms = Compose([Resize(mode="trilinear", align_corners=False,
                                        spatial_size=(96, 96, 96))
##                                        spatial_size=(90, 90, 90))
                                        ])
        tensor = train_transforms(tensor)

        return tensor

    def __getitem__(self, index):
        data_ct_rate = True
        if data_ct_rate:
            nii_file, input_text, onehotlabels = self.samples[index]
        else:
            nii_file, input_text = self.samples[index]
            onehotlabels = 0
        video_tensor = self.nii_to_tensor(nii_file)
        input_text = input_text.replace('"', '')  
        input_text = input_text.replace('\'', '')  
        input_text = input_text.replace('(', '')  
        input_text = input_text.replace(')', '')  
        name_acc = nii_file.split("/")[-2]
        return video_tensor, input_text, onehotlabels, name_acc
