from pprint import pprint
from modules.chexbert import CheXbert
import pandas as pd
import numpy as np


"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

label_set = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]

chexbert = CheXbert('/home/chenzhixuan/Workspace/MRG_baseline/checkpoints/stanford/chexbert/chexbert.pth', 'cpu').to('cpu')

train_data = pd.read_csv('/home/chenzhixuan/Workspace/LLM4CTRG/src/Dataset/my_data_csv/CTRG_train.csv')
val_data = pd.read_csv('/home/chenzhixuan/Workspace/LLM4CTRG/src/Dataset/my_data_csv/CTRG_val.csv')
test_data = pd.read_csv('/home/chenzhixuan/Workspace/LLM4CTRG/src/Dataset/my_data_csv/CTRG_test.csv')

# 合并数据框
data_df = pd.concat([train_data, val_data, test_data], ignore_index=True)

report_type = 'finding'
gt_list = data_df[report_type].tolist()

def mini_batch(gts, mbatch_size=16):
        length = len(gts)
        assert length == len(gts)
        for i in range(0, length, mbatch_size):
            yield gts[i:min(i + mbatch_size, length)]

def compute(gts):
    gts_chexbert = []
    for gt in mini_batch(gts):
        gt_chexbert = chexbert(list(gt)).tolist()
        gts_chexbert += gt_chexbert

    gts_chexbert = np.array(gts_chexbert)

    return gts_chexbert

gts_chexbert = compute(gt_list)

# combine gts_chexbert and id
gts_chexbert_df = pd.DataFrame(gts_chexbert, columns=label_set)
gts_chexbert_df['id'] = data_df['id']

gts_chexbert_df.to_csv('/home/chenzhixuan/Workspace/M2KT/data_csv/CTRG_finding_labels.csv', index=False)
