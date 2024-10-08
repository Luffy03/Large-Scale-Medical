<div align="center">
<h1>Large-Scale 3D Medical Image Pre-training</h1>

<a href="https://arxiv.org/abs/2402.17300"><img src='https://img.shields.io/badge/arXiv-Preprint-red' alt='Paper PDF'></a>
<a href="https://openaccess.thecvf.com/content/CVPR2024/html/Wu_VoCo_A_Simple-yet-Effective_Volume_Contrastive_Learning_Framework_for_3D_Medical_CVPR_2024_paper.html"><img src='https://img.shields.io/badge/CVPR-Conference-red' alt='Paper PDF'></a>
<a href='https://github.com/Luffy03/Large-Scale-Medical'><img src='https://img.shields.io/badge/Project_Page-VoCo-green' alt='Project Page'></a>
<a href='https://huggingface.co/Luffy503/VoCo/tree/main'><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Models-blue'></a>
<a href='https://huggingface.co/datasets/Luffy503/PreCT-160K'><img src='https://img.shields.io/badge/Dataset-PreCT--160K-pink' alt='Dataset'></a>
<a href='https://huggingface.co/datasets/Luffy503/VoComni'><img src='https://img.shields.io/badge/Dataset-VoComni-pink' alt='Dataset'></a>
</div>

This work presents VoCo, a new method for Large-Scale 3D Medical Image Pre-training. We release a new benchmark, including **160K** CT volumes (42M slices) for pre-training, **31M~1.2B** params of pre-trained models, and **50+** downstream tasks implementation.

![teaser](assets/data.png)

## News

- **2024-10-14:** Paper, code, models, and datasets are all released.

## Pre-trained Models

We provide various models for fine-tuning downstream tasks. For nnUNet, please refer to our nnunet trainer.

| Model           | Params |                                           Checkpoint                                           |
|:----------------|-------:|:----------------------------------------------------------------------------------------------:|
| VoComni_nnunet  |    31M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_nnunet.pt?download=true)  |
| VoCo_B_SSL_head |      - | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_B_SSL_head.pt?download=true) |
| VoCo_L_SSL_head |      - | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_L_SSL_head.pt?download=true) |
| VoCo_H_SSL_head |      - | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_H_SSL_head.pt?download=true) |
| VoComni_B       |    72M |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_B.pt?download=true)    |
| VoComni_L       |   290M |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_L.pt?download=true)    |
| VoComni_H       |   1.2B |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_H.pt?download=true)    |


### Load Pre-trained models
```python
import torch
import argparse
from monai.networks.nets import SwinUNETR

def load(model, state_dict):
    # make sure you load our checkpoints
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
args = parser.parse_args()
model = SwinUNETR(in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=args.feature_size,
        use_v2=True)
pretrained_path = ''
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

Please refer to [Acknowledgment](#Important). Download our pre-processed [downstream datasets](https://huggingface.co/datasets/Luffy503/VoCo_Downstream) for downstream tasks.

### Implementations
Please refer to [Downstream](./Downstream).

## Pre-training

### Download Pre-training Dataset

Please refer to [Acknowledgment](#Important). Download our  [PreCT-160K](https://huggingface.co/datasets/Luffy503/PreCT-160K) for pre-training.

### Various Pre-training recipes

Please refer to:

- [Self-supervised learning](./Self-supervised).

- [Semi-supervised learning](./Semi-supervised). 

- [Omni-supervised learning](./Omni-supervised). 



## VoComni Dataset

Please refer to [VoComni](./VoComni).


## Acknowledgement <a name="Important"></a>

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
