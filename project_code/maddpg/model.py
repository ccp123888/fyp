import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class ActorNetwork(nn.Module):

    def __init__(self, input_size, output_size, fc1_size=128, fc2_size=64, fc3_size=32):

        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, fc1_size, bias=True)
        self.fc2 = nn.Linear(fc1_size, fc2_size, bias=True)
        self.fc3 = nn.Linear(fc2_size, fc3_size, bias=True)
        self.fc4 = nn.Linear(fc3_size, output_size, bias=True)

        stdv1 = 1. / np.sqrt(fc1_size)
        stdv2 = 1. / np.sqrt(fc2_size)
        stdv3 = 1. / np.sqrt(fc3_size)

        self.fc1.weight.data.uniform_(-stdv1, stdv1)
        self.fc2.weight.data.uniform_(-stdv2, stdv2)
        self.fc3.weight.data.uniform_(-stdv3, stdv3)
        self.fc4.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, observation):

        input = observation

        x = F.relu(self.fc1(input))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        output = torch.tanh(self.fc4(x))

        return output

class CriticNetwork(nn.Module):

    def __init__(self, input_size, output_size, fc1_size=128, fc2_size=64, fc3_size=32):

        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(input_size, fc1_size, bias=True)
        self.fc2 = nn.Linear(fc1_size, fc2_size)
        self.fc3 = nn.Linear(fc2_size, fc3_size)
        self.fc4 = nn.Linear(fc3_size, output_size)

        stdv1 = 1. / np.sqrt(fc1_size)
        stdv2 = 1. / np.sqrt(fc2_size)
        stdv3 = 1. / np.sqrt(fc3_size)

        self.fc1.weight.data.uniform_(-stdv1, stdv1)
        self.fc2.weight.data.uniform_(-stdv2, stdv2)
        self.fc3.weight.data.uniform_(-stdv3, stdv3)
        self.fc4.weight.data.uniform_(-1e-3, 1e-3)

    def forward(self, observation, action):

        input = [observation, action]

        x = torch.cat(input, dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        output = self.fc4(x)

        return output
