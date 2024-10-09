<div align="center">
<h1>Large-Scale 3D Medical Image Pre-training</h1>

<a href="https://github.com/Luffy03/Large-Scale-Medical"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Wu_VoCo_A_Simple-yet-Effective_Volume_Contrastive_Learning_Framework_for_3D_Medical_CVPR_2024_paper.html"><img src='https://img.shields.io/badge/CVPR-Conference-red' alt='Paper PDF'></a>
<a href='https://github.com/Luffy03/Large-Scale-Medical'><img src='https://img.shields.io/badge/Project_Page-VoCo-green' alt='Project Page'></a>
<a href='https://huggingface.co/Luffy503/VoCo/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/Luffy503/PreCT-160K'><img src='https://img.shields.io/badge/Dataset-PreCT--160K-pink' alt='Dataset'></a>
<a href='https://huggingface.co/datasets/Luffy503/VoComni'><img src='https://img.shields.io/badge/Dataset-VoComni-pink' alt='Dataset'></a>
</div>

This work presents VoCo, a new method for Large-Scale 3D Medical Image Pre-training. We release a new benchmark, including **160K** CT volumes (**42M** slices) for pre-training, **31M~1.2B** params of pre-trained models, and **50+** downstream tasks implementation.

![teaser](assets/data.png)

[//]: # (## News)

[//]: # ()
[//]: # (- **2024-10-14:** Paper, code, models, and datasets are all released.)

## Quick start

- **[Models](https://huggingface.co/Luffy503/VoCo/tree/main):** **31M~1.2B** params of pre-trained models.
- **[Downstream](Downstream):** **50+** downstream tasks implementation, across segmentation, classification, registration, and vision-language.
- **[PreCT-160k](https://huggingface.co/datasets/Luffy503/PreCT-160K):** **160K** CT volumes (**42M** slices) for pre-training
- **[VoComni](https://huggingface.co/datasets/Luffy503/VoComni):** **20K** CT volumes with pseudo labels (20 organ & tumor classes)
- **[Self-supervised](Self-supervised):** Pre-training with unlabeled data
- **[Omni-supervised](Omni-supervised):** Pre-training with labeled and unlabeled data

## Pre-trained Models

We provide various models for fine-tuning downstream tasks. For nnUNet, please refer to [nnunet trainer](./Downstream/nnUNet).

- SSL_head represents trained by [Self-supervised pre-training](./Self-supervised).
- Omni represents trained by [Omni-supervised pre-training](./Omni-supervised). 

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
parser.add_argument("--out_channels", default=2, type=int, help="number of output channels")
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
**50+** downstream tasks implementations (updating). Please refer to [Downstream](./Downstream).

## Pre-training <a name="Pre-training"></a>

### Download Pre-training Dataset

Please refer to [Acknowledgment](#Acknowledgment). Download our  [PreCT-160K](https://huggingface.co/datasets/Luffy503/PreCT-160K) for pre-training.

**WARNING**: 
- It requires **22.6 TB** space to store the original datasets. For pre-training, it requires **30 TB** more space to cache the data, otherwise the pre-training will be very slow. And please store them in SSD.
- If you do not have enough space for PreCT-160K, you can try our [VoComni dataset](https://huggingface.co/datasets/Luffy503/VoComni). It requires less than **10 TB** only (?).

### Various Pre-training recipes

Please refer to:

- [Self-supervised pre-training](./Self-supervised/).

- [Omni-supervised pre-training](./Omni-supervised/). 


## VoComni Dataset

To facilitate the following research, we use VoCo to generate pseudo labels on **20K volumes**, with 20 foreground classes. Please refer to [VoComni](./VoComni).


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
