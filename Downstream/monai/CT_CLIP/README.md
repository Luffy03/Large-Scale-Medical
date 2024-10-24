**Modify pre-trained path** in sh files:
- swin_pretrained
- pretrained
- data-folder
- reports-file
- labels

Please follow the settings of [CT_CLIP](https://github.com/ibrahimethemhamamci/CT-CLIP).
You need to download BertTokenizer and BertModel models form their original repo.
```
microsoft/BiomedVLP-CXR-BERT-specialized
```

**Vocabulary classification Training** 
```
sh vocabfine_train_base.sh
sh vocabfine_train_large.sh
sh vocabfine_train_huge.sh
```
**Vocabulary classification Validation** 
```
python run_zero_shot.py
# modify path in run_zero_shot.py
inference = CTClipInference(
    clip,
    data_folder = '/CT-RATE/valid_preprocessed/',
    reports_file = "/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
    labels = "/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
    batch_size = 1,
    results_folder="./exps/",    # The path to save results
    num_train_steps = 1,
)
```

**retrieval validation**
```
python run_zero_shot.py
```
