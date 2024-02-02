import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class SampleDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
        self.len = self.X.shape[0]

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return self.len


def process_data(file_path):
    P = np.loadtxt(file_path,delimiter=',')
    Y = P.T
    X = Y.copy()
    X[X>0] = 1
    Y = Y/Y.sum(axis=1).reshape(-1,1)
    X = X/X.sum(axis=1).reshape(-1,1)
    
    Y = Y.astype(np.float32)
    X = X.astype(np.float32)

    return X, Y


def get_data_loaders(file_path, batch_size, test_size):
    X, Y = process_data(file_path)

    if test_size > 0.0:
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=test_size)

        train_dataset = SampleDataset(X_train, Y_train)
        val_dataset = SampleDataset(X_val, Y_val)

        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=True)
    
    else:
        train_dataset = SampleDataset(X, Y)
        train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = None

    num_features = X.shape[1]

    return train_loader, val_loader, num_features