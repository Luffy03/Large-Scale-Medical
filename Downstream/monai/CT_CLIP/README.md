**Modify pre-trained path** in sh files:
- swin_pretrained
- pretrained
- data-folder
- reports-file
- labels

Please follow the settings of [CT_CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP).

**vocabulary classification**
```
sh vocabfine_train_base.sh
sh vocabfine_train_large.sh
sh vocabfine_train_huge.sh
```
**retrieval**
```
python run_zero_shot.py
```
