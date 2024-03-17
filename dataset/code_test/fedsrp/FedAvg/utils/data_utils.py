import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset, DataLoader

# Assuming hydro_data is a module that contains the HydroDataSetUnit17 class
from .hydro import *

'''def read_data(dataset_path, kind='train'):
    # Create an instance of the HydroDataSetUnit17 class
    hydro_dataset = HydroDataSetUnit17()

    # Load the data using the load method from the HydroDataSetUnit17 class
    if kind == 'train':
        train_x, train_y, _, _ = hydro_dataset.load()
        return {'x': train_x, 'y': train_y}

    else:
        _, _, test_x, test_y = hydro_dataset.load()
        return {'x': test_x, 'y': test_y}

def read_client_data(dataset_path, idx, task, is_train=True):
    # Read the data using the read_data function
    data = read_data(dataset_path, kind='train' if is_train else 'test')
    # Convert the data to PyTorch tensors
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    # Return the data as a dictionary
    return {'x': X, 'y': y}

def load_train_data(dataset_path, idx, task, batch_size=16):
    train_data = read_client_data(dataset_path, idx, task, is_train=True)
    train_dataset = TensorDataset(train_data['x'], train_data['y'])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)
    for i, (x, y) in enumerate(train_loader):
        print(f"Batch {i}: x - {x.shape}, y - {y.shape}")
        # 检查x的维度是否为3，如果不是则抛出错误
        if x.dim() != 3:
            raise ValueError(f"Expected input x to have 3 dimensions, got {x.dim()}")
    return DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

def load_test_data(dataset_path, idx, batch_size=16):
    test_data = read_client_data(dataset_path, idx, task=0, is_train=False)
    test_dataset = TensorDataset(test_data['x'], test_data['y'])
    return DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False)
import matplotlib.pyplot as plt
import pandas as pd

def read_data(kind='train'):
    # Create an instance of the HydroDataSetUnit17 class
    hydro_dataset = HydroDataSetUnit17()

    # Load the data using the load method from the HydroDataSetUnit17 class
    if kind == 'train':
        train_x, train_y, _, _ = hydro_dataset.load()
        
        # Print some content of train_x and train_y
        print(f"Sample of train_x:\n{train_x[:5]}")
        print(f"Sample of train_y:\n{train_y[:5]}")

        # Convert the data to PyTorch tensors
        train_x = torch.Tensor(train_x).type(torch.float32)
        train_y = torch.Tensor(train_y).type(torch.int64)
        
        # Convert train_y to a Pandas Series
        train_y = pd.Series(train_y)
        
        # Check label distribution
        plt.hist(train_y)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Label Distribution')
        plt.show()
        
        # Check label consistency
        grouped_data = train_y.groupby(train_y)
        for label, group in grouped_data:
            if len(group) > 1:
                print(f'Warning: Duplicate data points with label {label} found.')
        
        # Check with expert or domain knowledge
        expert_feedback = input('Is the label distribution and consistency correct? (y/n) ')
        if expert_feedback.lower() == 'n':
            print('Warning: Label distribution or consistency issue identified by expert.')

        return {'x': train_x, 'y': train_y}

    else:
        _, _, test_x, test_y = hydro_dataset.load()
        
        # Print some content of test_x and test_y
        print(f"Sample of test_x:\n{test_x[:5]}")
        print(f"Sample of test_y:\n{test_y[:5]}")

        # Convert the data to PyTorch tensors
        test_x = torch.Tensor(test_x).type(torch.float32)
        test_y = torch.Tensor(test_y).type(torch.int64)
        
        # Convert test_y to a Pandas Series
        test_y = pd.Series(test_y)
        
        # Check label distribution
        plt.hist(test_y)
        plt.xlabel('Label')
        plt.ylabel('Count')
        plt.title('Label Distribution')
        plt.show()
        
        # Check label consistency
        grouped_data = test_y.groupby(test_y)
        for label, group in grouped_data:
            if len(group) > 1:
                print(f'Warning: Duplicate data points with label {label} found.')
        
        return {'x': test_x, 'y': test_y}'''
def read_data(kind='train'):
    # Create an instance of the HydroDataSetUnit17 class
    hydro_dataset = HydroDataSetUnit17()

    # Load the data using the load method from the HydroDataSetUnit17 class
    if kind == 'train':
        train_x, train_y, _, _ = hydro_dataset.load()
        return {'x': train_x, 'y': train_y}

    else:
        _, _, test_x, test_y = hydro_dataset.load()
        return {'x': test_x, 'y': test_y}


def read_client_data(idx, task, is_train=True):
    # Read the data using the read_data function
    data = read_data(kind='train' if is_train else 'test')
    # Convert the data to PyTorch tensors
    X = torch.Tensor(data['x']).type(torch.float32)
    y = torch.Tensor(data['y']).type(torch.int64)
    # Return the data as a dictionary
    return {'x': X, 'y': y}

def load_train_data(idx, task, batch_size=16):
    train_data = read_client_data(idx, task, is_train=True)
    train_dataset = TensorDataset(train_data['x'], train_data['y'])
    return DataLoader(train_dataset, batch_size=batch_size, drop_last=False, shuffle=True)

def load_test_data(idx, batch_size=16):
    test_data = read_client_data(idx, task=0, is_train=False)
    test_dataset = TensorDataset(test_data['x'], test_data['y'])
    return DataLoader(test_dataset, batch_size=batch_size, drop_last=False, shuffle=False)



# Example usage
if __name__ == '__main__':
    dataset_path = './dataset/series_data/'
    train_data = read_client_data(dataset_path, idx=0, task=0, is_train=True)
    test_data = read_client_data(dataset_path, idx=0, task=0, is_train=False)

    print(train_data['x'].shape)
    print(train_data['y'].shape)
    print(test_data['x'].shape)
    print(test_data['y'].shape)
