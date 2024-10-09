# Save the paths of dataset dirs.
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'luna_ncs':
            return '../data/ncs_data'

        elif dataset == 'luna_ncc':
            return "/jhcnas1/wulinshan/Data/LUNA_Classification/MG_split/"

        elif dataset == 'luna_mg_pretask' or dataset == 'luna_ae_pretask':
            return "../../Data/Self_Learning_Cubes_1.0/bat_32_s_64x64x32"

        elif dataset == 'luna_pcrl_pretask' or dataset == 'luna_cl_pretask' or dataset == 'luna_pcrl_pretask_org':
            return "../data/PCRL_Cubes"

        elif dataset == 'luna_jigsaw_pretask' or dataset == 'luna_rkb_pretask' or dataset == 'luna_rkbp_pretask':
            return "/jhcnas1/wulinshan/Medical-SSL-master/LUNA2016_processed"# "../LUNA2016_processed"

        elif dataset == 'luna_rpl_pretask'  or dataset == 'luna_rot_pretask' \
                or dataset == 'luna_rpl_pretask_v2' or dataset == 'luna_rot_pretask_v2':
            return '../../Data/LUNA2016_cropped_x320y320z74'#'../../Data/Self_Learning_Cubes_3.0/bat_4_s_240x240x32'#'../../Data/LUNA2016_cropped_xyz'

        elif dataset == 'lits_seg_train' or dataset == 'lits_seg_test' \
                or dataset == 'lits_seg_liver_tumor_train' or dataset == 'lits_seg_liver_tumor_test':
            return '../data/LITS_liver_seg_org'

        elif dataset == 'btcv_seg_train' or dataset == 'btcv_seg_test':
            return '../data/Dataset100_BTCV/BTCV_processed'


        elif dataset == 'msd_liver_seg_train' or dataset == 'msd_liver_seg_test' or dataset == 'msd_liver_seg_train_down2' or dataset == 'msd_liver_seg_test_down2':
            return '../data/MSD_liver_seg'#'../../nnUNet/DATASET/nnUNet_preprocessed/Task003_Liver/nnUNetData_plans_v2.1_stage1'


        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError


