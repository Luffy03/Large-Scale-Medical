<div align="center">
<h1>Semi-supervised Learning is a Scalable Learner</h1>

</div>

Semi-supervised learning is the simplest way to unleash the power of labeled and unlabeled data simultaneously. Here, we provide a simple baseline for implementing large-scale semi-supervised pre-training.


## Pre-training

### Pre-trained Models

If you don't want to train from scratch, you can use our pre-trained models.

| Model           | Params |                                           Checkpoint                                           |
|:----------------|-------:|:----------------------------------------------------------------------------------------------:|
| VoComni_nnunet  |    31M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_nnunet.pt?download=true)  |
| VoCo_B_SSL_head |    53M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_B_SSL_head.pt?download=true) |
| VoCo_L_SSL_head |   206M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_L_SSL_head.pt?download=true) |
| VoCo_H_SSL_head |   818M | [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoCo_H_SSL_head.pt?download=true) |
| VoComni_B       |    72M |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_B.pt?download=true)    |
| VoComni_L       |   290M |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_L.pt?download=true)    |
| VoComni_H       |   1.2B |    [Download](https://huggingface.co/Luffy503/VoCo/resolve/main/VoComni_H.pt?download=true)    |

### Dataset
You can download our [VoComni](https://huggingface.co/datasets/Luffy503/VoComni) and assign them as labeled sets. For unlabeled sets, you can aggregate different sources of datasets into "imagesUn". It should be with same classes as [VoComni.json](./VoComni.json), or you can define by yourself. Here, we only provide a baseline for training.

The path should be organized as:
```
├── Data
    ├── imagesTr
    ├── labelsTr
    └── imagesUn
```

Use [gen_json.py](gen_json.py) to obtain "dataset_unlabeled.json".

### Usage

```bash
cd Semi-supervised
source activate YOUR-CONDA-ENVIRONMENT
# single GPU, if you don't have enough gpu resource
sh single_train
# multi-gpu
sh dist_B.sh
sh dist_L.sh
sh dist_H.sh
```

## Citation

If you find this repo useful for your research, please consider citing the paper as follows:

```bibtex
@article{wu2024large,
  title={Large-Scale 3D Medical Image Pre-training with Geometric Context Priors},
  author={Wu, Linshan and Zhuang, Jiaxin and Chen, Hao},
  journal={arXiv preprint arXiv:2410.09890},
  year={2024}
}
@InProceedings{voco-v1,
    author    = {Wu, Linshan and Zhuang, Jiaxin and Chen, Hao},
    title     = {VoCo: A Simple-yet-Effective Volume Contrastive Learning Framework for 3D Medical Image Analysis},
    booktitle = {CVPR},
    month     = {June},
    year      = {2024},
    pages     = {22873-22882}
}
```
