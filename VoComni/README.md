<div align="center">
<h1>VoComni Dataset: Fully-supervised Pre-training</h1>

<a href='https://huggingface.co/datasets/Luffy503/VoComni'><img src='https://img.shields.io/badge/Dataset-VoComni-pink' alt='Dataset'></a>
</div>

We use our pre-trained models to generate pseudo labels on **20K** volumes, with 20 organ & tumor classes. We provide a simple fully-supervised pre-training baseline for this [**VoComni**](https://huggingface.co/datasets/Luffy503/VoComni) dataset. **We find that without any complex designs, this supervised pre-training baseline can already outperform previous methods**.

**NOTE THAT** the generated pseudo labels of VoComni dataset are with **noises**, it can be also used for semi-supervised, weakly-supervised, or active learning research. The description is in [VoComni.json](VoComni.json), some cases are shown below.

![teaser](assets/vocomni.png)

## Pre-trained Models

If you do not want to train from scratch, you can choose to resume our pre-trained models for pre-training.

| Model           | Params |                                           Checkpoint                                           |
|:----------------|-------:|:----------------------------------------------------------------------------------------------:|
| VoCo_B_SSL_head |    53M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_B_SSL_head.pt?download=true) |
| VoCo_L_SSL_head |   206M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_L_SSL_head.pt?download=true) |
| VoCo_H_SSL_head |   818M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_H_SSL_head.pt?download=true) |
| VoComni_B       |    72M |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_B.pt?download=true)    |
| VoComni_L       |   290M |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_L.pt?download=true)    |
| VoComni_H       |   1.2B |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_H.pt?download=true)    |

## Download Dataset

Download our [VoComni](https://huggingface.co/datasets/Luffy503/VoComni) for pre-training.

We organize the path like [nnUNet](https://github.com/MIC-DKFZ/nnUNet), so you can also use [nnUNet](https://github.com/MIC-DKFZ/nnUNet) for training. I specify the name as Dataset503, since 503 is my name (◕‿◕✿).

The path of VoComni should be organized as:
```
├── VoComni
    ├── imagesTr
        ├── VoComni_0_0000.nii.gz
        ├──...
    ├── labelsTr
        ├── VoComni_0.nii.gz
        └──...
For nnUNet
├── Dataset503_VoComni
    ├── imagesTr
        ├── VoComni_0_0000.nii.gz
        ├──...
    ├── labelsTr
        ├── VoComni_0.nii.gz
        ├──...
    └── dataset.json
```

**WARNING**: 
- It requires about **10 TB** space to store and cache the datasets. 

## Pre-training

It is very simple to implement the pre-training
```
cd VoComni
source activate YOUR-CONDA-ENVIRONMENT
sh omni_B.sh
sh omni_L.sh
sh omni_H.sh
```

You need to modify the parameters:
- **pretrained_root**: The path you store the pretrained models
- **logdir**: The path you want to save your results
- **feature_size**: 48 Base (B), 96 Large (L), 192 Huge (H)
- **data_dir**: The path you store your dataset
- **cache_dir**: The path you want to cache your dataset (activated by 'use_persistent_dataset')
- **use_ssl_pretrained**: If True, use 'VoCo_SSL_head'. Else, 'VoComni'
- **use_persistent_dataset**: If True, it would cache data in 'cache_dir' for fast training. **WARNING**: it requires extra storage space !!!!!

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
