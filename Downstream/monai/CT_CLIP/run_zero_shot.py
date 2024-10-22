import torch
from transformer_maskgit import CTViT
from transformers import BertTokenizer, BertModel
from ct_clip import CTCLIP
from zero_shot import CTClipInference
import accelerate
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '5'

tokenizer = BertTokenizer.from_pretrained('microsoft/BiomedVLP-CXR-BERT-specialized',do_lower_case=True)
text_encoder = BertModel.from_pretrained("microsoft/BiomedVLP-CXR-BERT-specialized")
text_encoder.resize_token_embeddings(len(tokenizer))

backbone = 'swin_unetr'


if backbone == 'ct_clip':

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
    clip.load("/scratch/medimgfmod/zchenhi/checkpoints/CT-CLIP/CT_CLIP_zeroshot.pt")

elif backbone == 'swin_unetr':

    from baseline.swin import Swin
    # feature_size = 48
    # checkpoint_path = "./exps_base/checkpoint_15000_epoch_9.pt"
    # feature_size = 96
    # checkpoint_path = "./exps_large/checkpoint_5000_epoch_8.pt"
    feature_size = 192
    checkpoint_path = "./exps_huge/checkpoint_15000_epoch_5.pt"

    swin = Swin(in_channels=1, feature_size=feature_size)

    clip = CTCLIP(
        image_encoder=swin, text_encoder=text_encoder,
        dim_image=20736, dim_text=768, dim_latent=768, #196608 20736
        extra_latent_projection=False, use_mlm=False,
        downsample_image_embeds=False, use_all_token_embeds=False
    )
    clip.load(checkpoint_path, strict=False)

### superpod project
# inference = CTClipInference(
#     clip,
#     data_folder = '/project/medimgfmod/zchenhi/data/CT-RATE/valid_preprocessed/',
#     reports_file = "/project/medimgfmod/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
#     labels = "/project/medimgfmod/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
#     batch_size = 1,
#     results_folder="./exps_base/",    #inference_zeroshot
#     num_train_steps = 1,
# )

inference = CTClipInference(
    clip,
    data_folder = '/jhcnas5/nixuefeng/CT-RATE/valid_preprocessed/',
    reports_file = "/jhcnas5/nixuefeng/CT-RATE/dataset/radiology_text_reports/validation_reports.csv",
    labels = "/jhcnas5/nixuefeng/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv",
    batch_size = 1,
    results_folder="./exps_huge/",    #inference_zeroshot
    num_train_steps = 1,
)

inference.infer()
