import torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, input_size, output_size, layers=[1000, 1000]):
        super(Net, self).__init__()
        self.modules_list = nn.ModuleList()

        prev_layer_size = input_size
        
        # Dynamically add hidden layers
        for i, layer_size in enumerate(layers):
            self.modules_list.append(nn.Linear(prev_layer_size, layer_size))
            # self.modules_list.append(nn.ReLU())
            prev_layer_size = layer_size

        self.modules_list.append(nn.Linear(prev_layer_size, output_size))


    def forward(self, x):
        x2 = x.clone()
        for module in self.modules_list:
            x2 = module(x2)
        # for every value of x that is zero, the corresponding value of x2 is set to zero
        x2 = torch.mul(x2, x)
        x2 = nn.Softmax(dim=1)(x2)
        return x2


class ODEFunc(torch.nn.Module):
    def __init__(self):
        super(ODEFunc, self).__init__()
        self.fcc1 = torch.nn.Linear(N, N)
        self.fcc2 = torch.nn.Linear(N, N)

    def forward(self, t, y):
        out = self.fcc1(y)
        out = self.fcc2(out)
        f = torch.matmul(torch.matmul(torch.ones(y.size(dim=1),1),y),torch.transpose(out,0,1))
        return torch.mul(y,out-torch.transpose(f,0,1))

