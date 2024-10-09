from torch.utils.data import Dataset


class ClassificationBase(Dataset):
    def __init__(self,
                 config,
                 base_dir,
                 flag='train',
                 ):
        super(ClassificationBase, self).__init__()
        self.flag = flag
        self.config = config
        self.num_classes = config.class_num
        self._base_dir = base_dir
        self.all_images = []
        self.all_labels = []

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, index):
       pass