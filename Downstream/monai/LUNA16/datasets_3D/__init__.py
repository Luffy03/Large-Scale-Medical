from datasets_3D.paths import Path
from torch.utils.data import DataLoader, WeightedRandomSampler
from datasets_3D.Seg import SegmentationBTCVTrainSet, SegmentationBTCVTestset, SegmentationLunaSet,\
    SegmentationLiTSTrainSet, SegmentationLiTSTestset, SegmentationMSDLIVERTrainSet, SegmentationMSDLIVERTestset
from datasets_3D.MG import MGLunaPretaskSet
from datasets_3D.PCRL import PCRLLunaPretaskSet_ORIG, PCRLLunaPretaskSet
from datasets_3D.Classification import ClassificationLUNASet
from datasets_3D.AE import AELunaPretaskSet
from datasets_3D.Rubik_cube import JigSawLunaPretaskSet,RKBLunaPretaskSet, RKBPLunaPretaskSet
from datasets_3D.CL import CLLunaPretaskSet
from datasets_3D.PTP import RotLunaPretaskSet, RPLLunaPretaskSet, RPLLunaPretaskSet_v2, RotLunaPretaskSet_v2

datasets_dict_3D = {
    'luna_cl_pretask': CLLunaPretaskSet,
    'luna_pcrl_pretask': PCRLLunaPretaskSet,
    'luna_pcrl_pretask_org': PCRLLunaPretaskSet_ORIG,
    'luna_mg_pretask': MGLunaPretaskSet,
    'luna_ae_pretask': AELunaPretaskSet,
    'luna_rot_pretask': RotLunaPretaskSet,
    'luna_rot_pretask_v2': RotLunaPretaskSet_v2,
    'luna_rpl_pretask': RPLLunaPretaskSet,
    'luna_rpl_pretask_v2': RPLLunaPretaskSet_v2,
    'luna_rkb_pretask': RKBLunaPretaskSet,
    'luna_rkbp_pretask': RKBPLunaPretaskSet,
    'luna_jigsaw_pretask': JigSawLunaPretaskSet,

    'luna_ncs': SegmentationLunaSet,
    'luna_ncc': ClassificationLUNASet,
    'lits_seg_train':SegmentationLiTSTrainSet,
    'lits_seg_test':SegmentationLiTSTestset,
    'lits_seg_liver_tumor_train':SegmentationLiTSTrainSet,
    'lits_seg_liver_tumor_test': SegmentationLiTSTestset,
    'msd_liver_seg_train': SegmentationMSDLIVERTrainSet,
    'msd_liver_seg_test': SegmentationMSDLIVERTestset,
    'msd_liver_seg_train_down2': SegmentationMSDLIVERTrainSet,
    'msd_liver_seg_test_down2': SegmentationMSDLIVERTestset,
    'btcv_seg_train': SegmentationBTCVTrainSet,
    'btcv_seg_test': SegmentationBTCVTestset,
}


def get_dataloder_3D(args, flag="train", drop_last=True):
    '''
    :return: the dataloader of special datasets
    '''

    if flag == "train":
        print('******Building training dataloder******')
        datasets_name = args.train_dataset
        assert datasets_name in datasets_dict_3D.keys(), "The dataset use {} is not exist ".format(datasets_name)
        root = Path.db_root_dir(datasets_name)
        dataset = datasets_dict_3D[datasets_name](config=args, base_dir=root, flag=flag)
        batch_size = args.train_batch
        shuffle = True
        num_workers = args.num_workers
        pin_memory = True
    else:
        print('******Building test dataloder******')
        datasets_name = args.eval_dataset
        assert datasets_name in datasets_dict_3D.keys(), "The dataset use {} is not exist ".format(datasets_name)
        root = Path.db_root_dir(datasets_name)
        dataset = datasets_dict_3D[datasets_name](config=args, base_dir=root, flag=flag)
        batch_size = args.val_batch
        shuffle = False
        #num_workers = args.num_workers
        num_workers = 0
        pin_memory = True

    data_loader = DataLoader(dataset=dataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            drop_last=drop_last)

    return dataset, data_loader



