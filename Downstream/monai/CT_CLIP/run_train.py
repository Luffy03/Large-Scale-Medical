import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP, TextTransformer
from CTCLIPTrainer import CTClipTrainer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)

text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")

print("---------")
print(tokenizer.pad_token_id)
print(tokenizer.mask_token_id)
print("-----------")


image_encoder = CTViT(
    dim = 512,
    codebook_size = 8192,
    image_size = 480,
    patch_size = 30,
    temporal_patch_size = 15,
    spatial_depth = 4,
    temporal_depth = 4,
    dim_head = 32,
    heads = 8
)


clip = CTCLIP(
    image_encoder = image_encoder,
    text_encoder = text_encoder,
    dim_image = 2097152,
    dim_text = 768,
    dim_latent = 512,
    extra_latent_projection = False,         # whether to use separate projections for text-to-image vs image-to-text comparisons (CLOOB)
    use_mlm=False,
    downsample_image_embeds = False,
    use_all_token_embeds = False

)
trainer = CTClipTrainer(
    clip,
    reports_file_train= "/project/medimgfmod/CT-RATE/dataset/radiology_text_reports/train_reports.csv",
    reports_file_valid= "/project/medimgfmod/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
    data_train= "/scratch/medimgfmod/zchenhi/data/CT-RATE/train_preprocessed",
    data_valid = "/scratch/medimgfmod/zchenhi/data/CT-RATE/valid_preprocessed",
    labels = "/project/medimgfmod/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv",
    batch_size = 8,
    results_folder="/scratch/medimgfmod/csexuefeng/CT-CLIP/exps/exps_pretrain_ct_clip_ctrg",
    num_train_steps = 100001,
    num_workers = 4,
)

trainer.train()
