#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6

import copy
import pickle
import numpy as np
import pandas as pd
import torch
import argparse

from utils.options import args_parser
#from utils.train_utils import get_data, get_model
from utils.train_utils import get_model
from models.Update import LocalUpdate
from models.test import test_img, test_img_local, test_img_local_all
import os
import pdb
from torch.utils.data import TensorDataset
# dataset_path = 'Fmnist-client100-dir0.1'

if __name__ == '__main__':
    # parse args
    args = args_parser()
    #dataset_path = args.datasetpath
    #dataset_path = "/cifar10/dir_0-dot-1/"
    # Seed
    # torch.manual_seed(args.seed)#seed=1
    # torch.cuda.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = True
    # np.random.seed(args.seed)
    
    args.device = torch.device('cuda:{}'.format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else 'cpu')

    '''if args.unbalanced:
        base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
            dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)
    else:
        base_dir = './save/{}/{}_num{}_C{}_le{}_bs{}_round{}_m{}_lr{}/{}/'.format(
            dataset_path, args.model, args.num_users, args.frac, args.local_ep, args.local_bs, args.epochs, args.momentum, args.lr, args.results_save)'''
    #algo_dir = 'fedavg'
    
    #if not os.path.exists(os.path.join(base_dir, algo_dir)):
     #   os.makedirs(os.path.join(base_dir, algo_dir), exist_ok=True)

    # build a global model
    args.model = 'lstm'
    args.input_size = 5  # 输入特征的维度
    args.hidden_size = 64  # 隐藏层的维度
    args.num_layers = 2  # LSTM堆叠的层数
    args.output_size = 1  # 分类任务的输出维度（如果有的话）
    args.regression_output_size = 1  # 回归任务的输出维度

    #args.num_classes = 10000
    net_glob = get_model(args)
    net_glob.train()

    # build local models
    net_local_list = []
    for user_idx in range(args.num_users):
        net_local_list.append(copy.deepcopy(net_glob))
    
    # training
    #results_save_path = os.path.join(base_dir, algo_dir, 'results.csv')

    loss_train = []
    net_best = None
    best_loss = None
    best_acc = None
    best_epoch = None

    lr = args.lr
    results = []
    
    for iter in range(args.epochs):
        w_glob = None
        loss_locals = []
        
        # Client Sampling
        m = max(int(args.frac * args.num_users), 1)
        idxs_users = np.random.choice(range(args.num_users), m, replace=False)
        # print("Round {}, lr: {:.6f}, {}".format(iter, lr, idxs_users))
        task = iter//20#每过20个轮次进行任务切换
        # Local Updates
        print(idxs_users)
        for idx in idxs_users:
            #数据集名字，序号
            local = LocalUpdate(args=args, idxs=idx, task = task)
            net_local = copy.deepcopy(net_local_list[idx])
            w_local, loss = local.train(net=net_local.to(args.device), lr=lr)
            loss_locals.append(copy.deepcopy(loss))

            if w_glob is None:
                w_glob = copy.deepcopy(w_local)
            else:
                for k in w_glob.keys():
                    w_glob[k] += w_local[k]
        
        # Aggregation
        for k in w_glob.keys():
            w_glob[k] = torch.div(w_glob[k], m)
        
        # Broadcast
        update_keys = list(w_glob.keys())
        w_glob = {k: v for k, v in w_glob.items() if k in update_keys}
        for user_idx in range(args.num_users):
            net_local_list[user_idx].load_state_dict(w_glob, strict=False)
        net_glob.load_state_dict(w_glob, strict=False)
        if (iter + 1) == 50:
            lr = 0.005
        elif (iter + 1) ==75:
            lr = 0.001

        # print loss
        #loss_avg = sum(loss_locals) / len(loss_locals)
        #loss_train.append(loss_avg)

        #if (iter + 1) % args.test_freq == 0:
            #acc_test, acc_test_var, loss_test = test_img_local_all(net_local_list, args, return_all=False)
                        
            #print('Round {:3d}, Average loss {:.3f}, Test loss {:.3f}, Test accuracy: {:.2f}'.format(
             #   iter, loss_avg, loss_test, acc_test))

            #if best_acc is None or acc_test > best_acc:
        net_best = copy.deepcopy(net_glob)
             #   best_acc = acc_test
                #best_epoch = iter
                
                #best_save_path = os.path.join(base_dir, algo_dir, 'best_model.pt')
        best_save_path = '/root/best_model.pt'
        torch.save(net_best.state_dict(), best_save_path)
                
#                 for user_idx in range(args.num_users):
#                     best_save_path = os.path.join(base_dir, algo_dir, 'best_local_{}.pt'.format(user_idx))
#                     torch.save(net_local_list[user_idx].state_dict(), best_save_path)

        #    results.append(np.array([iter, loss_avg, loss_test, acc_test, best_acc]))
        #    final_results = np.array(results)
        #    final_results = pd.DataFrame(final_results, columns=['epoch', 'loss_avg', 'loss_test', 'acc_test', 'best_acc'])
        #    final_results.to_csv(results_save_path, index=False)

    #print('Best model, iter: {}, acc: {}'.format(best_epoch, best_acc))