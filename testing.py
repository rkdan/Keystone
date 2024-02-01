from types import SimpleNamespace

from keystone.model import Net
from main import train

training_params = SimpleNamespace(
        file_path = './data/DKI_data/Ptrain.csv',
        batch_size = 32,
        test_size = 0.0,
        lr = 0.001,
        epochs = 100,
        verbosity = 10,
        layers = [1000]
    )

import torch

state_dict = torch.load('./simple_net.pt')

net = Net(100, 100, layers=[1000])
net.load_state_dict(state_dict)

from keystone.data_loading import process_data
from keystone.keystone import calculate_keystone_array

X, Y = process_data('./data/DKI_data/Ptrain.csv')
K = calculate_keystone_array(X, Y, net)