import csv
from metrics import compute_scores

# 定义两个空字典，用于存储提取的信息
gt_dict = {}
pred_dict = {}

# def read_csv_to_dict(filename):
#     result_dict = {}
#     with open(filename, 'r') as file:
#         reader = csv.DictReader(file)
#         for row in reader:
#             id_value = row['ID']
#             ground_truth = row['Ground Truth']
#             result_dict[id_value] = [ground_truth]
#     return result_dict

# # 用法示例
# csv_filename = '/home/chenzhixuan/Workspace/LLM4CTRG/results/vit3d-radfm_perceiver-radfm_llama2-7b-chat-hf_ctr.csv'  # 替换为您的CSV文件名
# temp_gt_dict = read_csv_to_dict(csv_filename)

# 打开CSV文件
with open('/home/chenzhixuan/Workspace/M2KT/results/ctrg/V12_resnet101_labelloss_rankloss_20240208-133326/best.csv', 'r') as file:
    # 创建CSV读取器
    reader = csv.reader(file)

    # 跳过标题行（如果有的话）
    next(reader)

    # 遍历每一行数据
    for i, row in enumerate(reader):
        # GT和Pred
        id_value = row[0]
        pred_value = row[1]
        gt_value = row[2]

        # 将ID作为键，GT作为值存储在gt_dict中
        gt_dict[id_value] = [gt_value]

        # 将ID作为键，Pred作为值存储在pred_dict中
        pred_dict[id_value] = [pred_value]

# 计算评价指标
metrics = compute_scores(gt_dict, pred_dict)
print(metrics)
