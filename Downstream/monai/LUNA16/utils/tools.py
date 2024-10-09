import torch
from torchvision.transforms import transforms
import numpy as np
from PIL import  Image
import os
from collections import OrderedDict
import logging
import shutil
import subprocess
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import SimpleITK as sitk
import json

_SUPPORTED_EXTENSIONS = ['.tif', '.jpg', '.png', '.nii', '.gz', '.npy']
# _SUPPORTED_EXTENSIONS = ['.nii', '.gz']

def get_image_paths(input_directory):
    """Gets PNG and TIF image paths within a given directory.

  Args:
    input_directory: String name of input directory, without trailing '/'.
    max_images: Integer, max number of images paths to return.

  Returns:
    List of strings of image paths.
  """
    # os.walk might require path to directory without trailing '/'.
    assert input_directory
    assert input_directory[-1] != '/'
    paths = []
    print('Searching %s for PNG, TIF, JPG and NII files.' % input_directory)

    for directory, _, files in os.walk(input_directory):
        for f in files:
            path = os.path.join(directory, f)
            if os.path.splitext(path)[1] in _SUPPORTED_EXTENSIONS:
            # if os.path.split(path)[1] == 'hr.png':
                paths.append(path)
    if not paths:
        print('No images found in directory.')
        return []
    return paths


def create_exp_dir(path, desc='Experiment dir: {}'):
    if not os.path.exists(path):
        os.makedirs(path)
    print(desc.format(path))


def convert_state_dict(state_dict):
    """Converts a state dict saved from a dataParallel module to normal
       module state_dict inplace
       :param state_dict is the loaded DataParallel model_state
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def get_logger(log_dir):
    create_exp_dir(log_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(log_dir, 'run.log'))
    fh.setFormatter(logging.Formatter(log_format))
    logger = logging.getLogger('Nas Seg')
    logger.addHandler(fh)
    return logger



class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def save_config(obj, path):
    with open(os.path.join(path, 'config.json'), 'w', encoding='utf-8') as fObj:
        json.dump(obj, fObj, cls=MyEncoder)

def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)

def get_gpus_memory_info():
    """Get the maximum free usage memory of gpu"""
    rst = subprocess.run('nvidia-smi -q -d Memory',stdout=subprocess.PIPE, shell=True).stdout.decode('utf-8')
    rst = rst.strip().split('\n')
    memory_available = [int(line.split(':')[1].split(' ')[1]) for line in rst if 'Free' in line][::2]
    id = int(np.argmax(memory_available))
    return id, memory_available

def calc_parameters_count(model):
    return np.sum(np.prod(v.size()) for v in model.parameters())/1e6


def save_network(state, epoch, checkpoint_path):
    model_out_path = checkpoint_path + "model_epoch{}.pth".format(epoch)
    # state = {'model': model.state_dict(), 'optimizer': self.optimizer.state_dict(), 'epoch': epoch}
    # check path status
    if not os.path.exists(checkpoint_path):
        os.makedirs(checkpoint_path)
    # save model
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


def load_continue_network(net, optimizer, latest_model):
    checkpoint = torch.load(latest_model)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print('Successfully save model {}！'.format(latest_model))
    start_epoch = checkpoint['epoch'] + 1
    print('Successfully load epoch {}！'.format(start_epoch))
    return start_epoch


def load_part_params(keywords, model, load_path, device):
    """
    keywords: a list of the layer to be loaded.
    """

    checkpoint = torch.load(load_path, map_location=device)
    model_pretrained = checkpoint['state_dict']
    model_dict = model.state_dict()
    layers = []
    for k, v in model_pretrained.items():
        for keyword in keywords:
            if k.find(keyword)!=-1:
                layers.append(k)

    pretrained_dict = {k: v for k, v in model_pretrained.items() if k in layers}
    # 更新现有的model_dict
    model_dict.update(pretrained_dict)

    # 加载我们真正需要的state_dict
    model.load_state_dict(model_dict)

    return model, checkpoint


def load_part_params_except(except_keywords, model, load_path, device):
    """
    keywords: a list of the layer to be loaded.
    """

    checkpoint = torch.load(load_path, map_location=device)
    model_pretrained = checkpoint['state_dict']
    model_dict = model.state_dict()
    layers = []
    for k, v in model_pretrained.items():
        for keyword in except_keywords:
            if k.find(keyword) == -1:
                layers.append(k)

    pretrained_dict = {k: v for k, v in model_pretrained.items() if k in layers}
    # update the model_dict
    model_dict.update(pretrained_dict)

    # load the state_dict
    model.load_state_dict(model_dict, strict=False)

    return model, checkpoint

def save_tensor2image(tensor, name, path):
    """
    Save a tensor image to the path.
    name: file_name
    path: file_path
    """

    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path + '/' + name + '.png'

    # Tensor to PIL.Image
    PIL = transforms.ToPILImage()(tensor.cpu())
    PIL.save(file_path)


def save_np2image(nimage, name, path):
    """
    Save a tensor image to the path.
    name: file_name
    path: file_path
    """

    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path + '/' + name + '.png'

    # Tensor to PIL.Image
    PIL = Image.fromarray(nimage*255).convert('L')
    PIL.save(file_path)


def save_tensor2heatmap(tensor, name, path, color="YlGnBu_r"):
    """
    Save a tensor image to the path.
    name: file_name
    path: file_path
    """

    if not os.path.exists(path):
        os.makedirs(path)
    file_path = path+ '/' + name +'_heatmap.png'

    # Tensor to PIL.Image
    PIL = transforms.ToPILImage()(tensor.cpu())

    # Heat map
    sns_plot = sns.heatmap(PIL,cmap=color)
    s1 = sns_plot.get_figure()
    s1.savefig(file_path, dpi=300, bbox_inches='tight')
    plt.close()


def save_tensor2nii(savedImg, saved_path, saved_name):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    savedImg = savedImg.cpu().numpy()
    newImg = sitk.GetImageFromArray(savedImg)
    sitk.WriteImage(newImg, os.path.join(saved_path, saved_name+'.nii'))


def save_np2nii(savedImg, saved_path, saved_name, direction=None):
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    newImg = sitk.GetImageFromArray(savedImg)
    if direction is not None:
        newImg.SetDirection(direction)
    sitk.WriteImage(newImg,  os.path.join(saved_path, saved_name+'.nii'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Aggregate_3DSeg_tool():
    """
    Aggregate all the patches in one sample for 3D segmentation testing.
    """
    def __init__(self, img_org_shape, img_new_shape, C, num_classes):
        self.result = None
        self.org_shape = img_org_shape
        self.new_shape = img_new_shape
        self.C = C
        self.num_classes = num_classes

    def add_patch_result(self, tensor):
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
        else:
            self.result = tensor

        # result [num_patches, K, D, H, W]

    def recompone_overlap(self):
        """
        prediction of the model shape：[N_patchs_img, K, patch_s, patch_h, patch_w]
        Return: result of recompone output: [K, img_s, img_h, img_w]
        """
        patch_d = self.result.shape[-3]
        patch_h = self.result.shape[-2]
        patch_w = self.result.shape[-1]
        #print('agg', self.result.shape, self.new_shape[1], self.new_shape[2], self.new_shape[3])
        N_patches_d = (self.new_shape[-3] - patch_d) // self.C['stride_d'] + 1
        N_patches_h = (self.new_shape[-2] - patch_h) // self.C['stride_h'] + 1
        N_patches_w = (self.new_shape[-1] - patch_w) // self.C['stride_w'] + 1
        N_patches_img = N_patches_d * N_patches_h * N_patches_w
        print("N_patches_s/h/w:", N_patches_d, N_patches_h, N_patches_w)
        print("N_patches_img: " + str(N_patches_img))
        assert (self.result.shape[0] == N_patches_img)

        full_prob = torch.zeros((self.num_classes, self.new_shape[-3], self.new_shape[-2], self.new_shape[-1]))
        full_sum = torch.zeros((self.num_classes, self.new_shape[-3], self.new_shape[-2], self.new_shape[-1]))
        k = 0
        for s in range(N_patches_d):
            for h in range(N_patches_h):
                for w in range(N_patches_w):
                    full_prob[:, s * self.C['stride_d']:s * self.C['stride_d'] + patch_d,
                    h * self.C['stride_h']:h * self.C['stride_h'] + patch_h,
                    w * self.C['stride_w']:w * self.C['stride_w'] + patch_w] += self.result[k]

                    a = full_prob[:, s * self.C['stride_d']:s * self.C['stride_d'] + patch_d,
                    h * self.C['stride_h']:h * self.C['stride_h'] + patch_h,
                    w * self.C['stride_w']:w * self.C['stride_w'] + patch_w]
                    # print('*****', a.size(), self.result[k].size())

                    full_sum[:, s * self.C['stride_d']:s * self.C['stride_d'] + patch_d,
                    h * self.C['stride_h']:h * self.C['stride_h'] + patch_h,
                    w * self.C['stride_w']:w * self.C['stride_w'] + patch_w] += 1
                    k += 1
        assert (k == self.result.size(0))
        assert (torch.min(full_sum) >= 1.0)
        final_avg = full_prob / full_sum
        #final_avg = full_prob / k
        assert (torch.max(final_avg) <= 1.0)
        assert (torch.min(final_avg) >= 0.0)
        aggregated_pred = final_avg[:, :self.org_shape[-3], :self.org_shape[-2], :self.org_shape[-1]]

        return aggregated_pred


def create_one_hot_label(num_classes, label):
    """
    Input label: [D, H, W].
    Output label: [K, D, H, W]. The output label contains the background class in the 0th channel.
    """

    onehot_label = np.zeros(
        (num_classes, label.shape[0], label.shape[1], label.shape[2]), dtype=np.int32)
    for i in range(num_classes):
        onehot_label[i, :, :, :] = (label == i).astype(np.int32)

    return onehot_label


def one_hot_reverse(one_hot_label):
    """
    Input label: [K, D, H, W].
    Output label: [D, H, W].
    """
    label = np.zeros((one_hot_label.shape[-3], one_hot_label.shape[-2], one_hot_label.shape[-1]))
    for i in range(one_hot_label.shape[0]):
        label[one_hot_label[i, :] == 1] = np.int32(i + 1)
    return label

def select_target_type(y, criterion):
    # convert type of target according to criterion
    if criterion in ['ce', 'kappa']:
        y = y.long()
    elif criterion in ['mse', 'mae', 'smooth_L1']:
        y = y.float()
    elif criterion in ['focal_loss']:
        y = y.to(dtype=torch.int64)
    else:
        raise NotImplementedError('Not implemented criterion.')
    return y


