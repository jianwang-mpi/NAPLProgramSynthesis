import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, n_state, n_action, hidden_num=128):
        super(Net, self).__init__()
        self.n_state = n_state
        self.n_action = n_action
        self.hidden_num = hidden_num

        self.fc1 = nn.Linear(self.n_state, self.hidden_num)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        self.fc2 = nn.Linear(self.hidden_num, self.hidden_num)
        self.fc2.weight.data.normal_(0, 0.1)  # initialization
        self.out = nn.Linear(self.hidden_num, self.n_action)
        self.out.weight.data.normal_(0, 0.1)   # initialization

    def forward(self, state):
        x = state
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        actions_value = self.out(x)
        return actions_value