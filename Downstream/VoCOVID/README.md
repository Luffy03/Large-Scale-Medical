<div align="center">
<h1>VoCovid: Semi-supervised Covid Segmentation</h1>
<a href='https://huggingface.co/datasets/Luffy503/VoCovid'><img src='https://img.shields.io/badge/Dataset-VoCovid-pink' alt='Dataset'></a>
</div>

We use VoCo to generate pseudo labels for semi-supervised covid segmentation, just for fun. If you are interested in this topic, you can follow the following instructions.

## Download downstream datasets

2871 volumes for covid segmentation. Download our pre-processed datasets from our [Hugging face](https://huggingface.co/datasets/Luffy503/VoCovid). 

```
├── VoCovid
    ├── imagesTr
        ├── VoCovid_00001_0000.nii.gz
        ├── ...
    ├── labelsTr
        ├── VoCovid_00001.nii.gz
        ├── ...
    ├── imagesUn
    ├── labelsUn
    ├── imagesVal
    └── labelsVal
```

## Usage
We provide a simple baseline for covid segmentation. **NOTE THAT** labels in 'labelsUn' are noisy. You can investigate to develop algorithms to overcome it

### Fine-tuning

```
cd VoCOVID
source activate YOUR-CONDA-ENVIRONMENT
sh train.sh
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
```