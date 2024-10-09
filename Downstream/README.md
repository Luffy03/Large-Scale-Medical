<div align="center">
<h1>Downstream</h1>
</div>

We release implementations of **50+** downstream datasets across various medical tasks, including segmentation, classification, registration, and vision-language. We will consistently update this repo to build a comprehensive validation benchmark for medical pre-training.

| Dataset                                     | Modality | Task                    |
|---------------------------------------------|----------|-------------------------|
| [BTCV](MONAI/BTCV/)                         | CT       | Abdomen Seg.            |
| [AMOS22](MONAI/Amos/)                       | CT       | Abdomen Seg.            |
| [WORD](MONAI/Word/)                         | CT       | Abdomen Seg.            |
| [FLARE22](MONAI/Flare22/)                   | CT       | Abdomen Seg.            |
| [FLARE23](MONAI/Flare22/)                   | CT       | Abdomen Seg.            |
| [Abdomenct1k](MONAI/Abdomen1k/)             | CT       | Abdomen Seg.            |
| [AbdomenAtlas](MONAI/AbdomenAtlas/)         | CT       | Abdomen Seg.            |
| [TotalSegmentator](MONAI/Totalsegmentator/) | CT       | 104 Structures Seg.     |
| [MM-WHS](MONAI/MM-WHS/)                     | CT       | Heart Seg.              |
| [ASOCA](MONAI/ASOCA/)                       | CT       | coronary Seg.           |
| [AVT](MONAI/Aorta/)                         | CT       | Aorta Seg.              |
| [CHAOS](MONAI/CHAOS/)                       | CT       | Liver Seg.              |
| [Sliver07](MONAI/Sliver07/)                 | CT       | Liver Seg.              |
| [IRCADb](MONAI/3D-IRCADb/)                  | CT       | Liver Tumor Seg.        |
| [KiTS](MONAI/KiTs/)                         | CT       | Kidney Tumor Seg.       |
| [KiPA22](MONAI/Kipa/)                       | CT       | Kidney Tumor Seg.       |
| [TCIA-Panc.](MONAI/TCIA_Panc/)              | CT       | Panc. Seg.              |
| [PANORAMA](MONAI/PANORAMA/)                 | CT       | Panc. Tumor Seg.        |
| [SegThor](MONAI/SegThor/)                   | CT       | Thoracic Risk Seg.      |
| [BHSD](MONAI/BHSD/)                         | CT       | Brain Bleed Seg.        |
| [StructSeg19](MONAI/StructSeg19/)           | CT       | Nasopharynx Cancer Seg. |
| [Verse20](nnUNet)                           | CT       | Vertebrae Seg.          |
| [PENGWIN](MONAI/PENGWIN)                    | CT       | Vertebrae Seg.          |
| [Covid-19-20](MONAI/COVID/)                 | CT       | Covid Seg.              |
| [FUMPE](MONAI/FUMPE/)                       | CT       | Pulmonary Embolism Seg. |
| [Parse22](MONAI/Parse22/)                   | CT       | Pulmonary Artery Seg.   |
| [AIIB23](MONAI/AIIB23/)                     | CT       | Fibrotic Lung Seg.      |
| [CC-CCII](MONAI/CC-CCII/)                   | CT       | Covid Classi.           |
| [LUNA16](MONAI/LUNA16/)                     | CT       | Lung Nodule Classi.     |
| [AutoPET-II23](nnUNet)                      | PET-CT   | HeadNeck Lesion Seg.    |
| [AMOS-MRI](MONAI/AMOS-MRI/)                 | MRI      | Abdomen Seg.            |
| [MM-WHS-MRI](MONAI/MM-WHS-MRI/)             | MRI      | Heart Seg.              |
| [ACDC](MONAI/ACDC/)                         | MRI      | Heart Seg.              |
| [ATLAS-MRI](MONAI/ATLAS-MRI/)               | MRI      | Liver Tumor Seg.        |
| [BraTs21](MONAI/BraTS21/)                   | MRI      | Brain Tumor Seg.        |
| [IXI](MONAI/Registration/)                  | MRI      | Brain MRI Registration  |
| [OASIS](MONAI/Registration/)                | MRI      | Brain MRI Registration  |
| [CTRG-Chest](MONAI/M2KT/)                   | VLP      | Report Generation       |
| [CT-RATE](MONAI/CT_CLIP/)                   | VLP      | Vocabulary Classi.      |
| [CT-RATE](MONAI/CT_CLIP/)                   | VLP      | Report-Volume Retrieval |
| MSD Challenge                               |          |                         |
| [Task01 Brain](MONAI/BraTS/)                | MRI      | Brain Tumor Seg.        |
| [Task02 Heart](MONAI/Heart/)                | MRI      | Heart Seg.              |
| [Task03 Liver](MONAI/LiTs/)                 | CT       | Liver Tumor Seg.        |
| [Task04 Hip.](MONAI/Hip/)                   | MRI      | Hip. Seg.               |
| [Task05 Pros.](MONAI/Prostate/)             | MRI      | Prostate Seg.           |
| [Task06 Lung](MONAI/Lung/)                  | CT       | Lung Cancer Seg.        |
| [Task07 Panc.](MONAI/Panc/)                 | CT       | Pancreas Tumor Seg.     |
| [Task08 Vessel](MONAI/Vessel/)              | CT       | Vessel Tumor Seg.       |
| [Task09 Spleen](MONAI/Spleen/)              | CT       | Spleen Seg.             |
| [Task10 Colon](MONAI/Colon/)                | CT       | Colon Cancer Seg.       |

## Download downstream datasets

**NOTE THAT** we are not the authors of these datasets. Although all these datasets are publicly available for academic research, you need to cite the original works as shown in our paper. For certain datasets (e.g., [WORD](https://github.com/HiLab-git/WORD)) that necessitate approval from the authors, you need to download it from the original link.

You can choose to download our pre-processed datasets from our [Hugging face](https://huggingface.co/datasets/Luffy503/VoCo_Downstream). Most of these datasets are organized like [nnUNet](https://github.com/MIC-DKFZ/nnUNet).

```
├── YOUR/DIRECTORY/OF/PRETRAINED/DATA
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
We provide both [MONAI](MONAI) and [nnUNet](nnUNet) implementations. For [nnUNet](nnUNet), you need to follow the instructions from their [official repo](https://github.com/MIC-DKFZ/nnUNet). 

### Fine-tuning
Here, we take [3D-IRCADb](./MONAI/3D-IRCADb) as an example:
```
cd 3D-IRCADb
source activate YOUR-CONDA-ENVIRONMENT
sh train.sh
```
A template for train.sh like:
```
now=$(date +"%Y%m%d_%H%M%S")
name=VoCo
logdir=runs/logs_swin_base_VoComni
feature_size=48
data_dir=/data/3Dircadb1_convert/
cache_dir=/data/cache/3D-IRCADb
use_ssl_pretrained=False
use_persistent_dataset=True

mkdir -p $logdir

torchrun --master_port=21503 main.py \
    --name $name \
    --feature_size $feature_size \
    --data_dir $data_dir \
    --cache_dir $cache_dir \
    --use_ssl_pretrained $use_ssl_pretrained \
    --use_persistent_dataset $use_persistent_dataset \
    --logdir $logdir | tee $logdir/$now.txt
```

**Parameters you need to modify !!!!!** :
- name: The name of pre-trained models. Support [VoCo, suprem, swin, clip_driven, mg, unimiss, dodnet] for now
- master_port: specify different master_port for different processes
- logdir: The path you want to save your results
- feature_size: 48 Base (B), 96 Large (L), 192 Huge (H)
- data_dir: The path you store your dataset
- cache_dir: The path you want to cache your dataset (activated by 'use_persistent_dataset')
- use_ssl_pretrained: If True, use 'VoCo_SSL_head'. Else, 'VoComni'
- use_persistent_dataset: If True, it would cache data in 'cache_dir' for fast training. **WARNING**: it requires extra storage space !!!!!

### Validation and Testing 

We provide template for [validation](./val.py) and [testing](./test.py). You should modify the parameters in these two files according to your own settings !!! We provide the descriptions of these parameters in the files.

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
### Classification

Please refer to [CC-CCII](MONAI/CC-CCII) and [LUNA16](MONAI/LUNA16). 

### Registration

Please refer to [Registration](MONAI/Registration). Please cite [TransMorph](https://github.com/junyuchen245/TransMorph_Transformer_for_Medical_Image_Registration) and follow their instructions.

### Vision-language

Please refer to [M2KT](MONAI/M2KT) and [CT_CLIP](MONAI/CT_CLIP). Please cite [CT_CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP) and follow their instructions.

## Downstream Checkpoints
We are uploading our fine-tuning checkpoints to [BaiduYun](https://github.com/Luffy03/Large-Scale-Medical/Downstream).

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
@article{Monai,
  title={Monai: An open-source framework for deep learning in healthcare},
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