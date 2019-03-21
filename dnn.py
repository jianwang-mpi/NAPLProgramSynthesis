import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from env.env import Env
from network.FullConnected import Net
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
from tensorboard_logger import configure, log_value
import time
import os
env = BitVector()
N_ACTIONS = env.n_actions
N_STATES = len(env.to_list()) * 2
actions = env.op.operations
log_dir = '../logs/{}'.format(time.time())
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
configure(logdir=log_dir)


class BitVectorDataset(Dataset):

    def __init__(self, path):
        self.data = []
        self.action_name_to_id = {}
        with open(path, 'rb') as f:
            self.raw_data = pickle.load(f)
        for i in range(len(actions)):
            action_name = actions[i]
            self.action_name_to_id[action_name] = i
        for item in tqdm(self.raw_data):
            input = item['input']
            target = item['target']
            solution = item['solution']
            env = BitVector(init=input, target=target)
            for action in solution:
                state = env.current_state()
                action_id = self.action_name_to_id[action]
                env.step(action_id)
                self.data.append({"state": state, "action": action_id})

    def __getitem__(self, index):
        item = self.data[index]
        state = item['state']
        action = item['action']
        x = torch.from_numpy(state).float()
        y = torch.zeros(N_ACTIONS).float()
        y[action] = 1.0
        return x, y

    def __len__(self):
        return len(self.data)


class Pretrain:
    def __init__(self, epoch=2000):
        self.net = Net(n_state=N_STATES, n_action=N_ACTIONS)
        self.dataset = BitVectorDataset('../dataset/bitvector_8bit_full.pkl')

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=16, shuffle=True, drop_last=True)
        self.epoch = epoch

        self.loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(params=self.net.parameters())

        self.test_dataloader = DataLoader(dataset=self.dataset, batch_size=1)
        self.max_acc = 0

    def train(self):
        self.net.train()
        for _ in range(self.epoch):
            loss_val = 0
            for x, y in tqdm(iter(self.dataloader)):
                y_ = self.net(x)
                self.optimizer.zero_grad()
                loss = self.loss(y_, y)
                loss.backward()
                self.optimizer.step()

                loss_val += loss.item()

            print("loss val is: {}".format(loss_val))
            log_value(name="loss", value=loss_val, step=_)

            self.test()

    def test(self, init_max=256, target_max=256, load_dir=None):
        self.net = self.net.eval()
        if load_dir is not None:
            self.net.load_state_dict(torch.load(load_dir))
        count = 0
        print('Start test')
        for init in tqdm(range(init_max)):
            for target in range(target_max):
                if init == target:
                    continue
                env = BitVector(init=init, target=target)
                s = env.current_state()
                ep_r = 0
                for i in range(8):
                    x = torch.unsqueeze(torch.FloatTensor(s), 0)
                    # input only one sample
                    actions_value = self.net(x)
                    action = torch.argmax(actions_value).item()
                    # step
                    s_, r, done, info = env.step(action)
                    ep_r += r
                    s = s_
                    if done:
                        break
                if ep_r > 0:
                    count += 1
        acc = float(count) / init_max / target_max
        if acc > self.max_acc:
            torch.save(self.net.state_dict(), '../models/pretrained.pkl')
            self.max_acc = acc
        print(acc)


def main():
    pretrain = Pretrain(30)
    # pretrain.net.load_state_dict(torch.load("../models/pretrained.pkl"))
    # pretrain.train()
    pretrain.test(init_max=1, load_dir="../models/pretrained.pkl")
    # pretrain.test()

if __name__ == '__main__':
    main()
