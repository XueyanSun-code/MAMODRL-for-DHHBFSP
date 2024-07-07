import torch
from torch import nn
import numpy as np

# D=[]  #记忆池
# N=1000   #容量
layers = 5
input = 7
output = 6
hidden = 30


class DQN_net(nn.Module):
    def __init__(self, input=7, output=6, hidden=30):
        super(DQN_net, self).__init__()
        self.input_layer = nn.Linear(input, hidden)
        self.relu=nn.ReLU()
        self.hidden_layer1 = nn.Linear(hidden, hidden)
        self.hidden_layer2=nn.Linear(hidden, hidden)
        self.hidden_layer3=nn.Linear(hidden, hidden)
        # self.hidden_layer4=nn.Linear(hidden, hidden)
        # self.hidden_layer5=nn.Linear(hidden, hidden)
        self.output_layer = nn.Linear(hidden, output)
        self.sf = nn.Softmax(dim=0)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.relu(x)
        x = self.hidden_layer1(x)
        x=self.relu(x)
        x = self.hidden_layer2(x)
        x = self.relu(x)
        x = self.hidden_layer3(x)
        x = self.relu(x)
        # x = self.hidden_layer4(x)
        # x = self.relu(x)
        # x = self.hidden_layer5(x)
        # x = self.relu(x)
        x = self.output_layer(x)
        x = self.sf(x)
        return x
