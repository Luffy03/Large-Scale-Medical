<div align="center">
<h1>Large-Scale 3D Medical Image Pre-training</h1>

<a href="https://github.com/Luffy03/Large-Scale-Medical"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Wu_VoCo_A_Simple-yet-Effective_Volume_Contrastive_Learning_Framework_for_3D_Medical_CVPR_2024_paper.html"><img src='https://img.shields.io/badge/CVPR-Conference-red' alt='Paper PDF'></a>
<a href='https://huggingface.co/Luffy503/VoCo/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/Luffy503/PreCT-160K'><img src='https://img.shields.io/badge/Dataset-PreCT--160K-green' alt='Dataset'></a>
<a href='https://huggingface.co/datasets/Luffy503/VoComni'><img src='https://img.shields.io/badge/Dataset-VoComni-green' alt='Dataset'></a>
<a href='https://huggingface.co/datasets/Luffy503/VoCovid'><img src='https://img.shields.io/badge/Dataset-VoCovid-green' alt='Dataset'></a>
</div>

This work presents **VoCo**, a new method for Large-Scale 3D Medical Image Pre-training. We release a new benchmark, including **160K** volumes (**42M** slices) for pre-training, **31M~1.2B** params of pre-trained models, various pre-training recipes, and **50+** downstream tasks implementation.

Linshan Wu, Jiaxin Zhuang, and <a href="https://scholar.google.com/citations?hl=en&user=Z_t5DjwAAAAJ">**Hao Chen**</a>. [**"Large-Scale 3D Medical Image Pre-training with Geometric Context Priors"**](./assets/main_paper.pdf). CVPR 2024 Extension.

![teaser](assets/data.png)

[//]: # (## News)

[//]: # ()
[//]: # (- **2024-10-14:** Paper, code, models, and datasets are all released.)

## Quick Start

- **[Models](https://huggingface.co/Luffy503/VoCo/tree/main):** **31M~1.2B** params of pre-trained models.
- **[Downstream](Downstream):** **50+** tasks implementations (segmentation, classification, registration, vision-language).
- **Datasets**:
- - **[PreCT-160K](https://huggingface.co/datasets/Luffy503/PreCT-160K):** **The existing largest dataset in this field: 160K** CT volumes (**42M** slices)
- - **[VoComni](https://huggingface.co/datasets/Luffy503/VoComni):** **20K** volumes with pseudo labels (20 organ & tumor classes)
- - **[VoCovid](/Downstream/VoCOVID):** Semi-supervised covid segmentation
- **Pre-training**:
- - **[Fully-supervised](VoComni):** Pre-training with labeled data
- - **[Self-supervised](Self-supervised):** Pre-training with unlabeled data
- - **[Semi-supervised](Semi-supervised):** Pre-training with labeled and unlabeled data
- - **[Omni-supervised](Omni-supervised):** Pre-training with labeled and unlabeled data

## Pre-trained Models

We provide various models for downstream tasks. For nnUNet, please refer to [nnunet trainer](./Downstream/nnUNet). If you want me to pre-train any other **advanced** networks, feel free to raise an issue and I will try to do it (^o^)/.

- 'SSL_head' represents trained by [Self-supervised pre-training](./Self-supervised).
- 'Omni' represents trained by [Omni-supervised pre-training](./Omni-supervised). 

| Model           | Params |                                           Checkpoint                                           |
|:----------------|-------:|:----------------------------------------------------------------------------------------------:|
| VoComni_nnunet  |    31M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_nnunet.pt?download=true)  |
| VoCo_B_SSL_head |    53M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_B_SSL_head.pt?download=true) |
| VoCo_L_SSL_head |   206M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_L_SSL_head.pt?download=true) |
| VoCo_H_SSL_head |   818M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_H_SSL_head.pt?download=true) |
| VoComni_B       |    72M |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_B.pt?download=true)    |
| VoComni_L       |   290M |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_L.pt?download=true)    |
| VoComni_H       |   1.2B |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_H.pt?download=true)    |

We download checkpoints of previous methods from [SuPreM](https://github.com/MrGiovanni/SuPreM) for comparison (Thanks for their great efforts!).

**Summary**: We spent over 10,000 GPU hours in evaluating 50+ downstream tasks. [SuPreM](https://github.com/MrGiovanni/SuPreM) appears to be the best in **previous methods**. You can try these models in [Downstream](./Downstream).

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
    ├── supervised_dodnet_unet_920.pth
    ├── supervised_clip_driven_universal_swin_unetr_2100.pth
    ├── self_supervised_unimiss_nnunet_small_5022.pth
    ├── self_supervised_nv_swin_unetr_5050.pt
    ├── self_supervised_models_genesis_unet_620.pt
    └── supervised_suprem_swinunetr_2100.pth
```

### Load Pre-trained models
```python
import torch
import argparse
from monai.networks.nets import SwinUNETR
def load(model, model_dict):
    # make sure you load our checkpoints
    if "state_dict" in model_dict.keys():
        state_dict = model_dict["state_dict"]
    else:
        state_dict = model_dict
    current_model_dict = model.state_dict()
    for k in current_model_dict.keys():
        if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()):
            print(k)
    new_state_dict = {
        k: state_dict[k] if (k in state_dict.keys()) and (state_dict[k].size() == current_model_dict[k].size()) else current_model_dict[k]
        for k in current_model_dict.keys()}
    model.load_state_dict(new_state_dict, strict=True)
    return model
parser = argparse.ArgumentParser(description="VoCo models")
parser.add_argument("--feature_size", default=48, type=int,
                    help="feature size: 48 Base (B), 96 Large (L), 192 Huge (H)")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=21, type=int, help="number of output channels")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
args = parser.parse_args()
model = SwinUNETR(img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_v2=True)
# YOUR PATH OF PRETRAINED MODELS. MODIFY IT
pretrained_path = './pretrained/VoComni_B.pt'
model_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
model = load(model, model_dict)
```
**NOTE**: "roi" is flexible according to your own settings. Your need to adjust "in_channels" and "out_channels" for specific datasets. If "in_channels != 1" or "out_channels != 21", only the first layer or the last layer would not be loaded.

## Fine-tuning

### Installation

```bash
git clone https://github.com/Luffy03/Large-Scale-Medical
cd Large-Scale-Medical
pip install -r requirements.txt
```

### Download Downstream Datasets

Please refer to [Acknowledgment](#Acknowledgment). Download our pre-processed [downstream datasets](https://huggingface.co/datasets/Luffy503/VoCo_Downstream) for downstream tasks.

### Implementations
Please refer to [Downstream](./Downstream): **50+** downstream tasks implementations.

**We are uploading our fine-tuning checkpoints to [BaiduYun](https://pan.baidu.com/s/1w75cJWoWfCt2FSjMDYl1FA?pwd=r1rp) to make sure fair comparisons**. A bit slow, so sorry for that!(╥﹏╥)

## Pre-training <a name="Pre-training"></a>

### Download Pre-training Dataset

Please refer to [Acknowledgment](#Acknowledgment). Download our  [PreCT-160K](https://huggingface.co/datasets/Luffy503/PreCT-160K) for pre-training.

**WARNING**: 
- It requires **22.6 TB** space to store the original datasets. For pre-training, it requires extra **30 TB** space to cache the data, otherwise the pre-training will be very slow. And please store them in SSD.
- If you do not have enough space for PreCT-160K, you can try our [VoComni dataset](https://huggingface.co/datasets/Luffy503/VoComni). It requires less than **10 TB** only (?).

### Various Pre-training recipes

Please refer to:

- [Fully-supervised pre-training](./VoComni/).
- [Self-supervised pre-training](./Self-supervised/).
- [Semi-supervised pre-training](./Semi-supervised).
- [Omni-supervised pre-training](./Omni-supervised/). 


## VoComni

To facilitate the following research, we use VoCo to generate pseudo labels on **20K volumes**, with 20 organ and tumor classes. Please refer to [VoComni](./VoComni).

## VoCovid
Please refer to [VoCovid](Downstream/VoCOVID) for **Semi-supervised Covid Segmentation**. Dataset can be downloaded from [hugging face](https://huggingface.co/datasets/Luffy503/VoCovid).

## Acknowledgement <a name="Acknowledgment"></a>

 **NOTE THAT** we are not the authors of these datasets. Although all these datasets are publicly available for academic research, you need to cite the original works as shown in our paper. For certain datasets (e.g., [WORD](https://github.com/HiLab-git/WORD)) that necessitate approval from the authors, you need to download it from the original link.

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
```
