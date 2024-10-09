from pprint import pprint

import pandas as pd
import json
import csv

from modules.metrics_clinical import CheXbertMetrics


# def main():
#     res_path = "/home/chenzhixuan/Workspace/R2Gen/results/swin3d-T3D_R2Gen/test_res_chexbert.csv"
#     gts_path = "/home/chenzhixuan/Workspace/R2Gen/results/swin3d-T3D_R2Gen/test_gts_chexbert.csv"
#     res_data, gts_data = pd.read_csv(res_path), pd.read_csv(gts_path)
#     res_data, gts_data = res_data.fillna(0), gts_data.fillna(0)

#     label_set = res_data.columns[1:].tolist()
#     res_data, gts_data = res_data.iloc[:, 1:].to_numpy(), gts_data.iloc[:, 1:].to_numpy()
#     res_data[res_data == -1] = 0
#     gts_data[gts_data == -1] = 0

#     metrics = compute_mlc(gts_data, res_data, label_set)
#     pprint(metrics)

def main():
    chexbert_metrics = CheXbertMetrics('/home/chenzhixuan/Workspace/MRG_baseline/checkpoints/stanford/chexbert/chexbert.pth', 16, 'cpu')
    
    # 读取 JSON 文件，假设文件名为data.json
    with open('/home/chenzhixuan/Workspace/M2KT/results/ctrg/V12_resnet101_labelloss_rankloss_20240208-133326/Enc2Dec-8_0_test_generated.json', 'r') as file:
        data = json.load(file)

    def read_csv_to_dict(filename):
        result_dict = {}
        with open(filename, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                id_value = row['ID']
                ground_truth = row['Ground Truth']
                result_dict[id_value] = ground_truth
        return result_dict

    # 用法示例
    csv_filename = '/home/chenzhixuan/Workspace/LLM4CTRG/results/vit3d-radfm_perceiver-radfm_llama2-7b-chat-hf_ctr.csv'  # 替换为您的CSV文件名
    temp_gt_dict = read_csv_to_dict(csv_filename)

    # 提取 "prediction" 和 "gt" 的值到列表
    id_list = [item['filename'] for item in data]
    res_list = [item['prediction'] for item in data]
    gt_list = [temp_gt_dict[item['filename']] for item in data]

    # 创建包含两个列表的字典
    data = {
        'ID': id_list,
        'Pred': res_list,
        'Ground_Truth': gt_list
    }

    # 创建 DataFrame
    df = pd.DataFrame(data)

    # 保存 DataFrame 到 CSV 文件
    df.to_csv('/home/chenzhixuan/Workspace/M2KT/results/ctrg/V12_resnet101_labelloss_rankloss_20240208-133326/best.csv', index=False)

    ce_scores = chexbert_metrics.compute(gt_list, res_list)
    for k, v in ce_scores.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
