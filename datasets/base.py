import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T


class BaseDataset(Dataset):
    def __init__(self, args, is_train=True, class_list=[]):
        self.dataset_path = args.data_path
        self.cropsize = args.crop_size
        self.is_train = is_train
        self.class_list = class_list

        self.x, self.y, self.mask, self.anomalies = self.load_dataset()

        self.transform_x = T.Compose([
            T.Resize(args.img_size, Image.ANTIALIAS),
            T.CenterCrop(args.crop_size),
            T.ToTensor()
        ])
        self.transform_mask = T.Compose([
            T.Resize(args.img_size, Image.NEAREST),
            T.CenterCrop(args.crop_size),
            T.ToTensor()])

        self.normalize = T.Compose([T.Normalize(args.norm_mean, args.norm_std)])

    def __getitem__(self, idx):
        img_path, y, mask, anomaly = self.x[idx], self.y[idx], self.mask[idx], self.anomalies[idx]
        
        x = Image.open(img_path).convert('RGB')
        x = self.normalize(self.transform_x(x))
        
        if y == 0:
            mask = torch.zeros([1, self.cropsize[0], self.cropsize[1]])
        else:
            mask = Image.open(mask)
            mask = self.transform_mask(mask)
        
        return x, y, mask, os.path.basename(img_path[:-4]), anomaly

    def __len__(self):
        return len(self.x)

    def load_dataset(self):
        phase = 'train' if self.is_train else 'test'
        x, y, mask, anomalies = [], [], [], []

        for class_name in self.class_list:
            img_dir = os.path.join(self.dataset_path, class_name, phase)
            gt_dir = os.path.join(self.dataset_path, class_name, 'ground_truth')

            anomaly_list = sorted(os.listdir(img_dir))
            for anomaly in anomaly_list:
                anomaly_img_dir = os.path.join(img_dir, anomaly)
                if not os.path.isdir(anomaly_img_dir):
                    continue
                img_list = sorted([os.path.join(anomaly_img_dir, f) for f in os.listdir(anomaly_img_dir)])
                x.extend(img_list)

                # load gt labels
                if anomaly == 'good':
                    y.extend([0] * len(img_list))
                    mask.extend([None] * len(img_list))
                    anomalies.extend(['good'] * len(img_list))
                else:
                    y.extend([1] * len(img_list))
                    anomaly_gt_dir = os.path.join(gt_dir, anomaly)
                    img_names = [os.path.splitext(os.path.basename(f))[0] for f in img_list]
                    gt_list = [os.path.join(anomaly_gt_dir, f + '_mask.png') for f in img_names]
                    mask.extend(gt_list)
                    anomalies.extend([anomaly] * len(img_list))

        return list(x), list(y), list(mask), list(anomalies)
