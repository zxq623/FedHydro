#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import math
import pdb
import copy
from torch.optim import Optimizer
from utils.data_utils import load_train_data, load_test_data
from torch.utils.data import DataLoader

class LocalUpdate(object):
    def __init__(self, args, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.MSELoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        epoch_loss = []
        
        if local_eps is None:
            local_eps = self.args.local_ep_pretrain if self.pretrain else self.args.local_ep
        
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)
                # 确保labels是二维的
                labels = labels.view(-1, 1)
                loss = self.loss_func(logits, labels.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            print(f"Epoch {iter+1}, Loss: {epoch_loss[-1]}")

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)


    

