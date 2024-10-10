<div align="center">
<h1>nnUNet YYDS</h1>
</div>

Although has been published in *Nature Methods* for a few years, [nnUNet](https://github.com/MIC-DKFZ/nnUNet) still stands out as a strong segmentation framework. We also build a [VoCo trainer](nnunetv2/training/nnUNetTrainer/nnUNetTrainer_pretrain.py) for nnUNet implementation. **However**, it is worth noting that our models are pre-trained with [monai](monai) and transfer to [nnUNet](https://github.com/MIC-DKFZ/nnUNet). **Thus, the pre-processing settings are not consistent between pre-training and finetuing**, which may hinder its performance. We will investigate how to pre-train with nnUNet in the future.

## Usage
For usage, you need to follow the clear instructions in [nnUNet](https://github.com/MIC-DKFZ/nnUNet). Here, we take [Dataset503_VoComni](https://huggingface.co/datasets/Luffy503/VoComni) as an example to show the implementation. **This can be seen as a fully-supervised pre-training baseline for nnUNet**.

### Pre-trained Models

| Model           | Params |                                           Checkpoint                                           |
|:----------------|-------:|:----------------------------------------------------------------------------------------------:|
| VoComni_nnunet  |    31M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_nnunet.pt?download=true)  |

You need to modify the path of pre-trained models in [nnUNetTrainer_pretrain.py](nnunetv2/training/nnUNetTrainer/nnUNetTrainer_pretrain.py).

### Download Datasets

Download the datasets from our [Hugging face](https://huggingface.co/datasets/Luffy503/VoCo_Downstream). Take [VoComni](https://huggingface.co/datasets/Luffy503/VoComni) as an example:
```
├── /nnunet_data/nnUNet_raw/Dataset503_VoComni
    ├── imagesTr
        ├── VoComni_0_0000.nii.gz
        ├──...
    ├── labelsTr
        ├── VoComni_0.nii.gz
        ├──...
    └── dataset.json
```


### Fine-tuning

```
cd nnUNet
source activate YOUR-CONDA-ENVIRONMENT
nnUNetv2_plan_and_preprocess -d 503 -c 3d_fullres --verbose --verify_dataset_integrity
nnUNetv2_train 503 3d_fullres all -tr nnUNetTrainer_pre
```

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