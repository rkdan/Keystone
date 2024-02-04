from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from keystone.data_loading import get_data_loaders
from keystone.model import Net


def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    return device


def train_epoch(net, train_loader, val_loader, loss_fn, optimizer, device):
    train_loss = 0.0
    val_loss = 0.0
    net.train()
    for X, Y in train_loader:
        X = X.to(device)
        Y = Y.to(device)
        optimizer.zero_grad()
        output = net(X)
        loss = loss_fn(output, Y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*X.size(0)
    train_loss = train_loss/len(train_loader.dataset)

    if val_loader is not None:
        net.eval()
        for X, Y in val_loader:
            X = X.to(device)
            Y = Y.to(device)
            output = net(X)
            loss = loss_fn(output, Y)
            val_loss += loss.item()*X.size(0)
        val_loss = val_loss/len(val_loader.dataset)
        
    else:
        val_loss = np.nan
    
    
    return train_loss, val_loss


def train(params):

    train_loader, val_loader, num_features = get_data_loaders(params.file_path, params.batch_size, params.val_size)

    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = nn.MSELoss()

    # kl divergence loss
    # loss_fn = nn.functional.kl_div()

    net = Net(num_features, num_features, layers=params.layers)
    optimizer = torch.optim.Adam(net.parameters(), lr=params.lr)

    # training loop
    device = get_device()
    net.to(device)
    best_model = net.state_dict()
    train_losses = []
    val_losses = []

    optimal_loss = 1
    for epoch in range(params.epochs):
        train_loss, val_loss = train_epoch(net, train_loader, val_loader, loss_fn, optimizer, device)
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)

        if val_loss < optimal_loss:
            optimal_loss = val_loss
            best_model = net.state_dict()

        if epoch % params.verbosity == 0:
            print('Epoch: {} \tTraining Loss: {:.2e} \tValidation Loss: {:.2e}'.format(epoch, train_loss, val_loss))


    # save best model
    torch.save(best_model, 'simple_net.pt')
    loss = {'train': train_losses, 'val': val_losses}
    torch.save(loss, 'simple_net_loss.pt')

    return loss, best_model, train_loader, val_loader

if __name__ == '__main__':
    training_params = SimpleNamespace(
        file_path = './data/Ptrain.csv',
        batch_size = 32,
        test_size = 0.0,
        lr = 0.001,
        epochs = 500,
        verbosity = 50
    )

    train(params)