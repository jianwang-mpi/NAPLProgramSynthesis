import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from env.env import Env
import env.action as action_def
from network.FullConnected import Net
import torch.nn.functional as F
import pickle
import numpy as np
from tqdm import tqdm
from tensorboard_logger import configure, log_value
import time
import os

env = Env()
N_STATES = env.n_state
log_dir = 'logs/{}'.format(time.time())
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
configure(logdir=log_dir)


class GrammarNet(torch.nn.Module):
    def forward(self, state):
        root_result = self.root_grammar(state)
        delete_node_reslt = self.delete_node_grammar(state)
        delete_edge_result = self.delete_edge_grammar(state)
        filter_result = self.filter_grammar(state)
        find_path_result = self.find_path_grammar(state)

        return root_result, [delete_node_reslt, delete_edge_result, filter_result, find_path_result]

    def __init__(self):
        super().__init__()
        self.root_grammar = Net(n_state=N_STATES, n_action=len(action_def.action))
        self.delete_node_grammar = Net(n_state=N_STATES, n_action=len(action_def.delete_node_action))
        self.delete_edge_grammar = Net(n_state=N_STATES, n_action=len(action_def.delete_edge_action))
        self.filter_grammar = Net(n_state=N_STATES, n_action=len(action_def.filter_action))
        self.find_path_grammar = Net(n_state=N_STATES, n_action=2 * len(action_def.find_path_source))


class BitVectorDataset(Dataset):
    def __init__(self, path):
        self.action_name_to_id = {}
        self.data = self.generate_dataset(path)

    def __getitem__(self, index):
        item = self.data[index]
        state = item['state']
        action = item['action']
        x = torch.from_numpy(state).float()
        root_action = action[0]
        y_root = torch.zeros(len(action_def.action)).float()
        y_root[root_action] = 1.0
        leaf_action = action[1]
        y_leaf = None
        if root_action == 0:
            y_leaf = torch.zeros(len(action_def.delete_node_action)).float()
            y_leaf[leaf_action] = 1.0
        elif root_action == 1:
            y_leaf = torch.zeros(len(action_def.delete_edge_action)).float()
            y_leaf[leaf_action] = 1.0
        elif root_action == 2:
            y_leaf = torch.zeros(len(action_def.filter_action)).float()
            y_leaf[leaf_action] = 1.0
        elif root_action == 3:
            y_leaf = torch.zeros(2 * len(action_def.find_path_source)).float()
            y_leaf[leaf_action[0]] = 1.0
            y_leaf[len(action_def.find_path_source) + leaf_action[1]] = 1.0
        return x, y_root, y_leaf

    def __len__(self):
        return len(self.data)

    def generate_dataset(self, raw_data_file_path):
        raw_data = np.load(raw_data_file_path)
        dataset = []
        for item in raw_data:
            state = item[0]
            init_target = state[int(len(state) / 2):]
            actions = item[1]
            env = Env(target_state=init_target)
            for action in actions:
                if action != 0:
                    dataset.append({"state": state, "action": action})
                state, reward, done = env.step(action[0], action[1])
                if done:
                    break
        return dataset


class Pretrain:
    def __init__(self, epoch=2000):
        self.net = GrammarNet()
        self.dataset = BitVectorDataset('dataset/raw_data.npy')

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=1, shuffle=True, drop_last=True)
        self.epoch = epoch

        self.loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(params=self.net.parameters())

        self.test_dataloader = DataLoader(dataset=self.dataset, batch_size=1)
        self.max_acc = 0

    def train(self):
        self.net.train()
        for _ in range(self.epoch):
            loss_root_val = 0
            loss_leaf_val = 0
            for x, y_root, y_leaf in tqdm(self.dataloader):
                y_root_, y_leafs = self.net(x)

                root_selection = torch.argmax(y_root).item()
                self.optimizer.zero_grad()
                loss_root = self.loss(y_root, y_root_)

                loss_root.backward()
                loss_leaf = self.loss(y_leaf, y_leafs[root_selection])
                loss_leaf.backward()
                self.optimizer.step()

                loss_root_val += loss_root.item()
                loss_leaf_val += loss_leaf.item()

            print("loss_root_val is: {}".format(loss_root_val))
            print("loss_leaf_val is: {}".format(loss_leaf_val))
            log_value(name="loss_root_val", value=loss_root_val, step=_)
            log_value(name="loss_leaf_val", value=loss_leaf_val, step=_)
            # self.test()

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
                env = Env()
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
    pretrain = Pretrain(300)
    # pretrain.net.load_state_dict(torch.load("../models/pretrained.pkl"))
    pretrain.train()
    # pretrain.test(init_max=1, load_dir="../models/pretrained.pkl")
    # pretrain.test()


if __name__ == '__main__':
    main()
