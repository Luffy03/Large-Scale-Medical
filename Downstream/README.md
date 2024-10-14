<div align="center">
<h1>Downstream Medical Tasks</h1>
<a href="https://github.com/Luffy03/Large-Scale-Medical"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Wu_VoCo_A_Simple-yet-Effective_Volume_Contrastive_Learning_Framework_for_3D_Medical_CVPR_2024_paper.html"><img src='https://img.shields.io/badge/CVPR-Conference-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/Luffy503/VoCo/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/Luffy503/VoCo_Downstream'><img src='https://img.shields.io/badge/Dataset-Downsteam-green' alt='Dataset'></a>
<a href='https://huggingface.co/datasets/Luffy503/VoCovid'><img src='https://img.shields.io/badge/Dataset-VoCovid-green' alt='Dataset'></a>
</div>

We release implementations of **50+** downstream datasets across various medical tasks, including segmentation, classification, registration, and vision-language. We will consistently update this repo to build a comprehensive validation benchmark for medical pre-training.

| Dataset                                     | Modality | Task                    |
|---------------------------------------------|----------|-------------------------|
| [BTCV](monai/BTCV/)                         | CT       | Abdomen Seg.            |
| [AMOS22](monai/Amos/)                       | CT       | Abdomen Seg.            |
| [WORD](monai/Word/)                         | CT       | Abdomen Seg.            |
| [FLARE22](monai/Flare22/)                   | CT       | Abdomen Seg.            |
| [FLARE23](monai/Flare22/)                   | CT       | Abdomen Seg.            |
| [Abdomenct1k](monai/Abdomen1k/)             | CT       | Abdomen Seg.            |
| [AbdomenAtlas](monai/AbdomenAtlas/)         | CT       | Abdomen Seg.            |
| [TotalSegmentator](monai/Totalsegmentator/) | CT       | 104 Structures Seg.     |
| [MM-WHS](monai/MM-WHS/)                     | CT       | Heart Seg.              |
| [ASOCA](monai/ASOCA/)                       | CT       | Coronary Seg.           |
| [AVT](monai/Aorta/)                         | CT       | Aorta Seg.              |
| [CHAOS](monai/CHAOS/)                       | CT       | Liver Seg.              |
| [Sliver07](monai/Sliver07/)                 | CT       | Liver Seg.              |
| [IRCADb](monai/3D-IRCADb/)                  | CT       | Liver Tumor Seg.        |
| [KiTS](monai/KiTs/)                         | CT       | Kidney Tumor Seg.       |
| [KiPA22](monai/Kipa/)                       | CT       | Kidney Tumor Seg.       |
| [TCIA-Panc.](monai/TCIA_Panc/)              | CT       | Panc. Seg.              |
| [PANORAMA](monai/PANORAMA/)                 | CT       | Panc. Tumor Seg.        |
| [SegThor](monai/SegThor/)                   | CT       | Thoracic Risk Seg.      |
| [BHSD](monai/BHSD/)                         | CT       | Brain Bleed Seg.        |
| [StructSeg19](monai/StructSeg19/)           | CT       | Nasopharynx Cancer Seg. |
| [Verse20](nnUNet)                           | CT       | Vertebrae Seg.          |
| [PENGWIN](monai/PENGWIN)                    | CT       | Vertebrae Seg.          |
| [Covid-19-20](monai/COVID/)                 | CT       | Covid Seg.              |
| [FUMPE](monai/FUMPE/)                       | CT       | Pulmonary Embolism Seg. |
| [Parse22](monai/Parse22/)                   | CT       | Pulmonary Artery Seg.   |
| [AIIB23](monai/AIIB23/)                     | CT       | Fibrotic Lung Seg.      |
| [CC-CCII](monai/CC-CCII/)                   | CT       | Covid Classi.           |
| [LUNA16](monai/LUNA16/)                     | CT       | Lung Nodule Classi.     |
| [AutoPET-II23](nnUNet)                      | PET-CT   | HeadNeck Lesion Seg.    |
| [AMOS-MRI](monai/AMOS-MRI/)                 | MRI      | Abdomen Seg.            |
| [MM-WHS-MRI](monai/MM-WHS-MRI/)             | MRI      | Heart Seg.              |
| [ACDC](monai/ACDC/)                         | MRI      | Heart Seg.              |
| [ATLAS-MRI](monai/ATLAS-MRI/)               | MRI      | Liver Tumor Seg.        |
| [BraTs21](monai/BRATS21/)                   | MRI      | Brain Tumor Seg.        |
| [IXI](monai/Registration/)                  | MRI      | Brain MRI Registration  |
| [OASIS](monai/Registration/)                | MRI      | Brain MRI Registration  |
| [CTRG-Chest](monai/M2KT/)                   | VLP      | Report Generation       |
| [CT-RATE](monai/CT_CLIP/)                   | VLP      | Vocabulary Classi.      |
| [CT-RATE](monai/CT_CLIP/)                   | VLP      | Report-Volume Retrieval |
| MSD Challenge                               |          |                         |
| [Task01 Brain](monai/BRATS/)                | MRI      | Brain Tumor Seg.        |
| [Task02 Heart](monai/Heart/)                | MRI      | Heart Seg.              |
| [Task03 Liver](monai/LiTs/)                 | CT       | Liver Tumor Seg.        |
| [Task04 Hip.](monai/Hip/)                   | MRI      | Hip. Seg.               |
| [Task05 Pros.](monai/Prostate/)             | MRI      | Prostate Seg.           |
| [Task06 Lung](monai/Lung/)                  | CT       | Lung Cancer Seg.        |
| [Task07 Panc.](monai/Panc/)                 | CT       | Pancreas Tumor Seg.     |
| [Task08 Vessel](monai/Vessel/)              | CT       | Vessel Tumor Seg.       |
| [Task09 Spleen](monai/Spleen/)              | CT       | Spleen Seg.             |
| [Task10 Colon](monai/Colon/)                | CT       | Colon Cancer Seg.       |

## Download downstream datasets

**NOTE THAT** we are not the authors of these datasets. Although all these datasets are publicly available for academic research, you need to cite the original works as shown in our paper. For certain datasets (e.g., [WORD](https://github.com/HiLab-git/WORD)) that necessitate approval from the authors, you need to download it from the original link.

You can choose to download our pre-processed datasets from our [Hugging face](https://huggingface.co/datasets/Luffy503/VoCo_Downstream). Most of these datasets are organized like [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

```
├── YOUR/DIRECTORY/OF/DOWNSTREAM/DATA
    ├── 3Dircadb1_convert
        ├── imagesTr
        ├── labelsTr
    ├── AIIB23
    ├── ATLAS-MRI
    ├── AVT
    ├── ...
    └── Segthor
```

## Usage
We provide both [monai](monai) and [nnUNet](nnUNet) implementations. For [nnUNet](nnUNet), you need to follow the instructions from their [official repo](https://github.com/MIC-DKFZ/nnUNet). 

### Pre-trained Models

We provide various models for fine-tuning downstream tasks. For nnUNet, please refer to [nnunet trainer](./Downstream/nnUNet).

- SSL_head represents trained by [Self-supervised pre-training](../Self-supervised).
- Omni represents trained by [Omni-supervised pre-training](../Omni-supervised). 

| Model           | Params |                                           Checkpoint                                           |
|:----------------|-------:|:----------------------------------------------------------------------------------------------:|
| VoComni_nnunet  |    31M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_nnunet.pt?download=true)  |
| VoCo_B_SSL_head |    53M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_B_SSL_head.pt?download=true) |
| VoCo_L_SSL_head |   206M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_L_SSL_head.pt?download=true) |
| VoCo_H_SSL_head |   818M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_H_SSL_head.pt?download=true) |
| VoComni_B       |    72M |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_B.pt?download=true)    |
| VoComni_L       |   290M |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_L.pt?download=true)    |
| VoComni_H       |   1.2B |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_H.pt?download=true)    |

We download checkpoints of other methods from [SuPreM](https://github.com/MrGiovanni/SuPreM) for comparison (Thanks for their great efforts!). 

The path of pre-trained models should be organized as:
```
├── YOUR/DIRECTORY/OF/PRETRAINED/MODELS
    ├── VoComni_nnunet.pt
    ├── VoCo_B_SSL_head.pt
    ├── VoCo_L_SSL_head.pt
    ├── VoCo_H_SSL_head.pt
    ├── VoComni_B.pt
    ├── VoComni_L.pt
    ├── VoComni_H.pt
    ├── ...
    └── supervised_suprem_swinunetr_2100.pth
```

### Fine-tuning
Here, we take [3D-IRCADb](./monai/3D-IRCADb) as an example:
```
cd 3D-IRCADb
source activate YOUR-CONDA-ENVIRONMENT
sh train.sh
```
A template for train.sh like:
```
now=$(date +"%Y%m%d_%H%M%S")
name=VoCo
pretrained_root=/pretrained
logdir=runs/logs_swin_base_VoComni
feature_size=48
data_dir=/data/3Dircadb1_convert/
cache_dir=/data/cache/3D-IRCADb
use_ssl_pretrained=False
use_persistent_dataset=True

mkdir -p $logdir

torchrun --master_port=21503 main.py \
    --name $name \
    --pretrained_root $pretrained_root \
    --feature_size $feature_size \
    --data_dir $data_dir \
    --cache_dir $cache_dir \
    --use_ssl_pretrained $use_ssl_pretrained \
    --use_persistent_dataset $use_persistent_dataset \
    --logdir $logdir | tee $logdir/$now.txt
```

**Parameters you need to modify !!!!!** :
- **name**: The name of pre-trained models. Support [VoCo, suprem, swin, clip_driven, mg, unimiss, dodnet] for now. **If None, without pre-training !!!**
- **pretrained_root**: The path you store the pretrained models
- **master_port**: specify different master_port for different processes
- **logdir**: The path you want to save your results
- **feature_size**: 48 Base (B), 96 Large (L), 192 Huge (H)
- **data_dir**: The path you store your dataset
- **cache_dir**: The path you want to cache your dataset (activated by 'use_persistent_dataset')
- **use_ssl_pretrained**: If True, use 'VoCo_SSL_head'. Else, 'VoComni'
- **use_persistent_dataset**: If True, it would cache data in 'cache_dir' for fast training. **WARNING**: it requires extra storage space !!!!!

### For Pre-processing Settings

We meticulously default settings for different downstream tasks, including "a_min, a_max, roi, spacing". We learn a lot from [nnUNet](https://github.com/MIC-DKFZ/nnUNet) and after consuming over 10,000 GPU hours in evaluation, we assume the current settings in 'main.py' are relatively better. 

**The settings may not be consistent with pre-training**, *e.g.*, 'roi=64' in pre-training while 'roi=96' in some downstream tasks. You can re-define these parameters yourself, but for fair comparisons, we recommend you to adopt our settings.


### Validation and Testing 

We provide template for [validation](./val.py) and [testing](./test.py). 

```
# Organized as 
├── Task/DIRECTORY
    ├── utils
        ├── utils.py
    ├── val.py
    ├── test.py
    └── ...

# val 
python val.py

# test
python test.py
```

You should modify the parameters in these two files according to your own settings !!! We provide the descriptions of these parameters in the files:
- **test_data_path**: The data path.
- **test_label_path**: The label path. Only in 'val.py'
- **trained_pth**: The model path.
- **input & output channels**: model settings
- **processing params**: consistent with training
- ......

### Classification

Please refer to [CC-CCII](monai/CC-CCII) and [LUNA16](monai/LUNA16). 

### Registration

Please refer to [Registration](monai/Registration). Please cite [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) and follow their instructions.

### Vision-language

Please refer to [M2KT](monai/M2KT) and [CT_CLIP](monai/CT_CLIP). Please cite [CT_CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) and follow their instructions.

## VoCovid
Please refer to [VoCovid](VoCOVID) for Semi-supervised Covid Segmentation. Dataset can be downloaded from [hugging face](https://huggingface.co/datasets/Luffy503/VoCovid).

## Downstream Checkpoints
We are uploading our fine-tuning checkpoints to [BaiduYun](https://pan.baidu.com/s/1w75cJWoWfCt2FSjMDYl1FA?pwd=r1rp). A bit slow, so sorry for that!(╥﹏╥)

## Citation

If you find this repo useful for your research, please consider citing the paper as follows:

```bibtex
@InProceedings{voco-v1,
    author    = {Wu, Linshan and Zhuang, Jiaxin and Chen, Hao},
    title     = {VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis},
    booktitle = {CVPR},
    month     = {June},
    year      = {2024},
    pages     = {22873-22882}
}
@article{monai,
  title={monai: An open-source framework for deep learning in healthcare},
  author={Cardoso, M Jorge and others},
  journal={arXiv preprint arXiv:2211.02701},
  year={2022}
}
@article{nnunet,
  title={nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation},
  author={Isensee, Fabian and others},
  journal={Nature Methods},
  volume={18},
  number={2},
  pages={203--211},
  year={2021},
}
```
