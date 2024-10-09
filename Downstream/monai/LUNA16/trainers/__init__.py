# Upstream/SSL pretext tasks
from trainers.mg_trainer import MGTrainer
from trainers.pcrl_trainer import PCRLTrainer
from trainers.pcrl3d_trainer import PCRL3DTrainer
from trainers.mypcrl3d_trainer import MYPCRL3DTrainer
from trainers.ssm_rot_trainer import RotTrainer
from trainers.ssm_rpl_trainer import RPLTrainer
from trainers.jigsaw_trainer import JigSawTrainer
from trainers.rkb_trainer import RKBTrainer
from trainers.rkb_plus_trainer import RKBPTrainer
from trainers.byol_trainer import BYOLTrainer
from trainers.simclr_trainer import SimCLRTrainer

# Downstream/target tasks
from trainers.seg3d_trainer import Seg3DTrainer # for simple one-object segmentation.
from trainers.seg3d_ROI_trainer import Seg3DROITrainer # for NCS, in which the ROI regions have been cropped.
from trainers.seg3d_mclasses_trainer import Seg3DMCTrainer # for multi-object segmentation (>=2)
from trainers.classification_trainer import ClassificationTrainer
from trainers.classification2d_trainer import Classification2DTrainer
from trainers.seg2d_trainer import Seg2DROITrainer
