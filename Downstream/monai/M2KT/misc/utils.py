#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2021/6/8
# @Author  : Shuxinyang
# @Contact : aspenstars@qq.com
# @FileName: utils.py

import logging
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn


def set_logging(logfile='expe_log.log'):
    logger = logging.getLogger()  # 不加名称设置root logger
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(
        '%(asctime)s-%(levelname)s:%(message)s',
        datefmt='%m-%d %H:%M:%S')

    # 使用FileHandler输出到文件
    fh = logging.FileHandler(logfile)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    # 使用StreamHandler输出到屏幕
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)

    # 添加两个Handler
    logger.addHandler(ch)
    logger.addHandler(fh)


def seed_everything(seed=0):
    # Setting the seed
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize(sample, maxval):
    """Scales images to be roughly [-1024 1024]."""

    if sample.max() > maxval:
        raise Exception("max image value ({}) higher than expected bound ({}).".format(sample.max(), maxval))

    sample = (2 * (sample.astype(np.float32) / maxval) - 1.) * 1024
    # sample = sample / np.std(sample)
    return sample
