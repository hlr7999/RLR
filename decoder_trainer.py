import math
import os
import time
import timm
import torch
import numpy as np
import torch.nn as nn

from datasets.mvtec import MVTecDataset, MVTEC_CLASS_NAMES
from models.decoder import Decoder
from models.embedding import Embedding2D

from timm.models.resnet import _cfg as res_cfg
from timm.models.efficientnet import _cfg as efn_cfg


class DecoderTrainer(object):
    def __init__(self, args):
        self.args = args
        
        kwargs = {'num_workers': args.num_workers, 'pin_memory': True}

        self.train_loaders = {}
        self.test_loaders = {}
        for c in MVTEC_CLASS_NAMES:
            train_dataset = MVTecDataset(args, is_train=True)
            self.train_loaders[c] = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=False, **kwargs)
            test_dataset  = MVTecDataset(args, is_train=False, class_name=c)
            self.test_loaders[c] = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, **kwargs)

        self.start_epoch = 0
        self.best_train_loss = 10000
        feat_dims = self.build_encoder()
        self.model = None
        self.l2_criterion = nn.MSELoss()
        if self.args.no_avg:
            self.save_dir = 'decoder_noavg'
        else:
            self.save_dir = 'decoder'
        if self.args.feature_levels == 3:
            self.save_dir += '_fl3'
    
    def build_encoder(self):
        if 'efficientnet' in self.args.backbone_arch:
            config = efn_cfg(url='', file=f'{self.args.root_path}/pretrained/tf_efficientnet_b6_aa-80ba17e4.pth')
        elif 'resnet50' in self.args.backbone_arch:
            config = res_cfg(url='', file=f'{self.args.root_path}/pretrained/wide_resnet50_racm-8234f177.pth')
        encoder = timm.create_model(
            self.args.backbone_arch,
            features_only=True,
            pretrained_cfg=config, 
            out_indices=self.args.out_indices,
            pretrained=True
        )
        self.encoder = encoder.to(self.args.device).eval()
        
        feat_dims = encoder.feature_info.channels()
        print("Feature Dimensions:", feat_dims)
        return feat_dims

    def build_model(self, c):
        self.start_epoch = 0
        self.best_train_loss = 10000
        self.model = Decoder(self.args.feature_levels)
        self.model.to(self.args.device)
        checkpoint = None
        path = os.path.join(self.args.root_path, self.save_dir, c, 'latest.pth')
        if os.path.exists(path):
            print('Resume..........')
            checkpoint = torch.load(path)
            state_dict = checkpoint['state_dict']
            self.model.load_state_dict(state_dict)
            self.start_epoch = checkpoint['epoch'] + 1
            self.best_train_loss = checkpoint['best_train_loss']
        print('Creating Models...Done')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr)
    
    def train(self):
        for c in MVTEC_CLASS_NAMES:
            print("======================================================")
            print(f"                      {c}")
            print("======================================================")
            self.build_model(c)
            self.train_one(c)

    def train_one(self, c):
        path = os.path.join(self.args.root_path, self.save_dir, c)
        if not os.path.exists(path):
            os.makedirs(path)

        train_loader = self.train_loaders[c]
        start_time = time.time()
        train_steps = len(train_loader)
        best_train_loss = self.best_train_loss

        for epoch in range(self.start_epoch, self.args.num_epochs):
            print("======================TRAIN MODE======================")
            iter_count = 0
            loss_list = []

            epoch_time = time.time()
            self.model.train()
            for i, (images, _, _, _, _) in enumerate(train_loader):
                iter_count += 1
                images = images.float().to(self.args.device)  # (N, 3, H, W)
                
                with torch.no_grad():
                    features = self.encoder(images)
                
                inputs = []
                for fl in range(self.args.feature_levels):
                    if self.args.no_avg:
                        input = features[fl]
                    else:
                        m = torch.nn.AvgPool2d(3, 1, 1)
                        input = m(features[fl])
                    inputs.append(input)

                output = self.model(inputs)
                loss = self.l2_criterion(output, images)

                loss_list.append(loss.item())

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            speed = (time.time() - start_time) / iter_count
            left_time = speed * ((self.args.num_epochs - epoch) * train_steps - i)
            print("Epoch: {} cost time: {}s | speed: {:.4f}s/iter | left time: {:.4f}s".format(epoch + 1, time.time() - epoch_time, speed, left_time))
            iter_count = 0
            start_time = time.time()

            train_loss = np.average(loss_list)
            print("Epoch: {0}, Steps: {1} | Rec Loss: {2:.7f}".format(epoch + 1, train_steps, train_loss))

            state = {
                'state_dict': self.model.state_dict(),
                'epoch': epoch,
                'best_train_loss': best_train_loss
            }
            if train_loss < best_train_loss:
                best_train_loss = train_loss
                state['best_train_loss'] = best_train_loss
                torch.save(state, os.path.join(path, 'best-train.pth'))
            torch.save(state, os.path.join(path, 'latest.pth'))
