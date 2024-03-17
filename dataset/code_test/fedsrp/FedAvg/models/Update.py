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


class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

    
'''class LocalUpdate(object):
    def __init__(self, args, idxs=None, task=0, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)#读取数据集
        self.ldr_train = load_train_data(idxs, task, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, lr, idx=-1, local_eps=None):
        net.train()

        # train and update
        
        # For ablation study
        optimizer = torch.optim.SGD(net.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
               # print(type(images))
                logits = net(images)
                loss = self.loss_func(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            print(f"Epoch {iter+1}, Loss: {epoch_loss[-1]}")

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)'''
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


    
class LocalUpdateBabu(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        # self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)#读取数据集
        self.ldr_train = load_train_data(dataset, idxs, batch_size=self.args.local_bs)
        self.pretrain = pretrain


    def train(self, net, body_lr, idx=-1, local_eps=None):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        
        optimizer = torch.optim.SGD(body_params, lr = body_lr, momentum=self.args.momentum, weight_decay=self.args.wd)

        epoch_loss = []
        
        if local_eps is None:
            if self.pretrain:
                local_eps = self.args.local_ep_pretrain
            else:
                local_eps = self.args.local_ep
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                logits = net(images)

                loss = self.loss_func(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

class LocalUpdateFedOur(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.sample_num = len(self.ldr_train)
        

    def train(self, net, lr, h_e=1):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': lr, 'name': "body"},
                                     {'params': head_params, 'lr': 0.0, "name": "head"}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        local_eps = self.args.local_ep
        
        for iter in range(local_eps):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)
                
                loss.backward()
                optimizer.step()
                
        
        for g in optimizer.param_groups:
            if g['name'] == "body":
                g['lr'] = 0.0
            elif g['name'] == 'head':
                g['lr'] = lr
        
        for iter in range(h_e):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)
                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()

        return net.state_dict()
    
class LocalUpdateFedOur_(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.sample_num = len(self.ldr_train)
        

    def train(self, net, lr, h_e=1):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD(body_params, lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        local_eps = self.args.local_ep
        
        for iter in range(local_eps):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                logits = net(images)
                loss = self.loss_func(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        optimizer = torch.optim.SGD(head_params, lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        
        for iter in range(h_e):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                logits = net(images)
                loss = self.loss_func(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return net.state_dict()

class LocalUpdateFedOur3(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.kl_loss = nn.KLDivLoss(reduction = 'batchmean')
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, batch_size=self.args.local_bs)
        self.pretrain = pretrain
        self.sample_num = len(self.ldr_train)
        

    def train(self, net, lr, per_weight):
        net.train()
        self.local_head = copy.deepcopy(net.linear)
        self.local_head.load_state_dict(per_weight, strict=True)
        self.local_head.train()
        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        for para in head_params:
            para.requried_grad = False

        optimizer = torch.optim.SGD(body_params, lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        local_eps = self.args.local_ep
        g_epoch_loss = []
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                logits = net(images)
                
                loss = self.loss_func(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_loss.append(loss.item())
            g_epoch_loss.append(sum(batch_loss)/len(batch_loss))
        
        for para in head_params:
            para.requried_grad = True
        for para in body_params:
            para.requried_grad = False
        
        optimizer = torch.optim.SGD(head_params, lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        optimizer_per = torch.optim.SGD(self.local_head.parameters(), lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)
        epoch_loss = []
        for iter in range(local_eps):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                rep = net.extract_features(images)
                per_logits = self.local_head(rep)
                loss = self.loss_func(per_logits, labels)
                optimizer_per.zero_grad()
                loss.backward()
                optimizer_per.step()
                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        # for iter in range(self.args.global_ep):
        for images, labels in self.ldr_train:
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            # rep = net.features(images)
            # rep = rep.view((rep.size(0), -1))
            rep = net.extract_features(images)
            global_logits = net.linear(rep)
            per_logits = self.local_head(rep.detach())
            y = F.softmax(per_logits,dim=1).detach()
            x_log = F.log_softmax(global_logits,dim=1)
            
            loss1 = self.loss_func(global_logits, labels)
            loss2 = self.kl_loss(x_log, y)
            loss = loss1 + loss2
            # print('loss2:',loss2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #返回全局模型参数，个性化模型参数，全局损失，个性化损失     
        return net.state_dict(), self.local_head.state_dict(), sum(g_epoch_loss)/len(g_epoch_loss), sum(epoch_loss) / len(epoch_loss)

class LocalUpdateFedOur2(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, lr, h_e=1):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': lr, 'name': "body"},
                                     {'params': head_params, 'lr': 0.0, "name": "head"}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        local_eps = self.args.local_ep
        
        for iter in range(local_eps):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                sample_per_class = torch.bincount(labels, minlength = self.args.num_classes).float()
                # zero_mask = sample_per_class == 0.0
                # sample_per_class[zero_mask] == 0.5
                weight_per_class = sample_per_class/sample_per_class.sum()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                out_y = net(images)
                # loss = self.loss_func(out_y, labels)
                loss = F.cross_entropy(out_y, labels, weight=weight_per_class)
                loss.backward()
                optimizer.step()
                
        for g in optimizer.param_groups:
            if g['name'] == "body":
                g['lr'] = 0.0
            elif g['name'] == 'head':
                g['lr'] = lr
        for iter in range(h_e):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                sample_per_class = torch.bincount(labels, minlength = self.args.num_classes).float()
                # zero_mask = sample_per_class == 0.0
                # sample_per_class[zero_mask] == 0.5
                weight_per_class = sample_per_class/sample_per_class.sum()
                # for index, x in enumerate(sample_per_class):
                #     if x >= 1.0:
                #         sample_per_class[index] = 1.0
                #     else:
                #         sample_per_class[index] = 0.0
                # weight_per_class = sample_per_class.float()
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                out_y = net(images)
                loss = F.cross_entropy(out_y, labels, weight=weight_per_class)
                loss.backward()
                optimizer.step()

        return net.state_dict()
    
def balanced_softmax_loss(labels, logits, sample_per_class, reduction="mean"):
    """Compute the Balanced Softmax Loss between `logits` and the ground truth `labels`.
    Args:
    labels: A int tensor of size [batch].
    logits: A float tensor of size [batch, no_of_classes].
    sample_per_class: A int tensor of size [no of classes].
    reduction: string. One of "none", "mean", "sum"
    Returns:
    loss: A float tensor. Balanced Softmax Loss.
    """
    spc = sample_per_class.type_as(logits)
    spc = spc.unsqueeze(0).expand(logits.shape[0], -1)
    logits = logits + spc.log()
    loss = F.cross_entropy(input=logits, target=labels, reduction=reduction)
    return loss
    
class LocalUpdatePerFedAvg(object):    
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)
        self.pretrain = pretrain

    def train(self, net, lr, beta=0.001, momentum=0.9):
        net.train()
        # train and update

        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        
        epoch_loss = []
        
        for local_ep in range(self.args.local_ep):
            batch_loss = []
            
            if len(self.ldr_train) / self.args.local_ep == 0:
                num_iter = int(len(self.ldr_train) / self.args.local_ep)
            else:
                num_iter = int(len(self.ldr_train) / self.args.local_ep) + 1
                
            train_loader_iter = iter(self.ldr_train)
            
            for batch_idx in range(num_iter):
                temp_net = copy.deepcopy(list(net.parameters()))
                    
                # Step 1
                for g in optimizer.param_groups:
                    g['lr'] = lr
                    
                try:
                    images, labels = next(train_loader_iter)
                except:
                    train_loader_iter = iter(self.ldr_train)
                    images, labels = next(train_loader_iter)
                    
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                
                net.zero_grad()
                
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
                
                
                # Step 2
                for g in optimizer.param_groups:
                    g['lr'] = beta
                    
                try:
                    images, labels = next(train_loader_iter)
                except:
                    train_loader_iter = iter(self.ldr_train)
                    images, labels = next(train_loader_iter)
                    
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                    
                net.zero_grad()
                
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                
                # restore the model parameters to the one before first update
                for old_p, new_p in zip(net.parameters(), temp_net):
                    old_p.data = new_p.data.clone()
                    
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))

        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 
    
    def one_sgd_step(self, net, lr, beta=0.001, momentum=0.9):
        net.train()
        # train and update

        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
        
        test_loader_iter = iter(self.ldr_train)

        # Step 1
        for g in optimizer.param_groups:
            g['lr'] = lr

        try:
            images, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(self.ldr_train)
            images, labels = next(train_loader_iter)


        images, labels = images.to(self.args.device), labels.to(self.args.device)

        net.zero_grad()

        logits = net(images)

        loss = self.loss_func(logits, labels)
        loss.backward()
        optimizer.step()

        # Step 2
        for g in optimizer.param_groups:
            g['lr'] = beta

        try:
            images, labels = next(train_loader_iter)
        except:
            train_loader_iter = iter(self.ldr_train)
            images, labels = next(train_loader_iter)

        images, labels = images.to(self.args.device), labels.to(self.args.device)

        net.zero_grad()

        logits = net(images)

        loss = self.loss_func(logits, labels)
        loss.backward()

        optimizer.step()


        return net.state_dict()

class LocalUpdateFedRep(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, lr):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': 0.0, 'name': "body"},
                                     {'params': head_params, 'lr': lr, "name": "head"}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        local_eps = self.args.local_ep
        
        for iter in range(local_eps):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
        
        for g in optimizer.param_groups:
            if g['name'] == "body":
                g['lr'] = lr
            elif g['name'] == 'head':
                g['lr'] = 0.0
        
        for batch_idx, (images, labels) in enumerate(self.ldr_train):
            images, labels = images.to(self.args.device), labels.to(self.args.device)
            net.zero_grad()
            logits = net(images)

            loss = self.loss_func(logits, labels)
            loss.backward()
            optimizer.step()

        return net.state_dict()


class LocalUpdatePredictor(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, lr):
        net.train()

        # train and update
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        for para in body_params:
            para.requried_grad = False

        optimizer = torch.optim.SGD(head_params, lr=lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        local_eps = self.args.local_ep
        
        for iter in range(local_eps):
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                loss.backward()
                optimizer.step()
        return net.state_dict()
    
class LocalUpdateFedProx(object):
    def __init__(self, args, dataset=None, idxs=None, pretrain=False):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
        self.ldr_train = load_train_data(dataset, idxs, batch_size=self.args.local_bs)
        self.pretrain = pretrain

    def train(self, net, body_lr, head_lr):
        net.train()
        g_net = copy.deepcopy(net)
        
        body_params = [p for name, p in net.named_parameters() if 'linear' not in name]
        head_params = [p for name, p in net.named_parameters() if 'linear' in name]
        
        optimizer = torch.optim.SGD([{'params': body_params, 'lr': body_lr},
                                     {'params': head_params, 'lr': head_lr}],
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.wd)

        epoch_loss = []
        
        for iter in range(self.args.local_ep):
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                net.zero_grad()
                logits = net(images)

                loss = self.loss_func(logits, labels)
                
                # for fedprox
                fed_prox_reg = 0.0
                for l_param, g_param in zip(net.parameters(), g_net.parameters()):
                    fed_prox_reg += (0.1 / 2 * torch.norm((l_param - g_param)) ** 2)
                loss += fed_prox_reg
                
                loss.backward()
                optimizer.step()

                batch_loss.append(loss.item())

            epoch_loss.append(sum(batch_loss)/len(batch_loss))
            
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss) 
    
    
class LocalUpdateDitto(object):
    def __init__(self, args, dataset=None, idxs=None):
        self.args = args
        self.loss_func = nn.CrossEntropyLoss()
        self.selected_clients = []
            
        self.ldr_train = DataLoader(DatasetSplit(dataset, idxs), batch_size=self.args.local_bs, shuffle=True)

    def train(self, net, w_ditto=None, lam=0, idx=-1, lr=0.1, last=False, momentum=0.9):
        net.train()
        # train and update
        bias_p=[]
        weight_p=[]
        for name, p in net.named_parameters():
            if 'bias' in name:
                bias_p += [p]
            else:
                weight_p += [p]
                
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)

        local_eps = self.args.local_ep
        args = self.args 
        epoch_loss=[]
        num_updates = 0
        
        for iter in range(local_eps):
            done=False
            batch_loss = []
            for batch_idx, (images, labels) in enumerate(self.ldr_train):
                w_0 = copy.deepcopy(net.state_dict())
                images, labels = images.to(self.args.device), labels.to(self.args.device)
                log_probs = net(images)
                loss = self.loss_func(log_probs, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if w_ditto is not None:
                    w_net = copy.deepcopy(net.state_dict())
                    for key in w_net.keys():
                        w_net[key] = w_net[key] - args.lr*lam*(w_0[key] - w_ditto[key])
                    net.load_state_dict(w_net)
                    optimizer.zero_grad()
                
                num_updates += 1
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss)/len(batch_loss))
        return net.state_dict(), sum(epoch_loss) / len(epoch_loss)

    
