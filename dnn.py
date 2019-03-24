import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from env.env import Env
import env.action as action_def
import numpy as np
from tqdm import tqdm
from tensorboard_logger import configure, log_value
import time
import os
from network.GrammarNet import GrammarNet


log_dir = 'logs/{}'.format(time.time())
if not os.path.exists(log_dir):
    os.mkdir(log_dir)
configure(logdir=log_dir)

class BitVectorDataset(Dataset):
    def __init__(self, path):
        self.action_name_to_id = {}
        # self.data = self.generate_dataset(path)
        self.data = self.generate_dataset_from_raw(path)

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

    def generate_dataset(self, data_file_path):
        data = np.load(data_file_path)
        dataset = []
        for item in data:
            state = item[0]
            action = item[1]
            dataset.append({"state": state, "action": action})
        return dataset

    def generate_dataset_from_raw(self, raw_data_file_path):
        raw_data = np.load(raw_data_file_path)
        dataset = []
        for item in raw_data:
            state = item[0]
            init_target = state[int(len(state) / 2):]
            actions = item[1]
            env = Env(target_state=init_target)
            for action in actions:
                dataset.append({"state": state, "action": action})
                state, reward, done = env.step(action[0], action[1])
                if done:
                    break
        return dataset


class Pretrain:
    def __init__(self, dataset='dataset/raw_data.npy', epoch=2000):
        self.net = GrammarNet()
        self.dataset = BitVectorDataset(dataset)

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
            self.test()

    def test(self, test_case_count=200, load_dir=None):
        self.net = self.net.eval()
        if load_dir is not None:
            self.net.load_state_dict(torch.load(load_dir))
        count = 0
        print('Start test')
        total_length = 0
        for _ in tqdm(range(test_case_count)):
            env = Env()
            s = env.get_current_state()
            ep_r = 0
            for i in range(4):
                x = torch.unsqueeze(torch.FloatTensor(s), 0)
                # input only one sample
                root_result, leaf_result = self.net(x)
                root_action = torch.argmax(root_result).item()
                if root_action != 3:
                    leaf_action = torch.argmax(leaf_result[root_action]).item()
                    # step
                    s_, r, done = env.step(root_action, leaf_action)
                else:
                    find_path_result = leaf_result[3]
                    find_path_source = torch.argmax(find_path_result[:, : int(find_path_result.shape[1] / 2)]).item()
                    find_path_target = torch.argmax(find_path_result[:, int(find_path_result.shape[1] / 2) :]).item()
                    # step
                    s_, r, done = env.step(root_action, (find_path_source, find_path_target))
                ep_r += r
                s = s_
                if done:
                    if ep_r > 0:
                        total_length += i
                    break
            if ep_r > 0:
                count += 1

        acc = float(count) / test_case_count
        if acc > self.max_acc and load_dir is None:
            torch.save(self.net.state_dict(), 'models/dnn.pkl')
            self.max_acc = acc
        print("acc is: ", acc)
        if count > 0:
            # 因为统计的时候少1，这里补上1
            print("length is: ", float(total_length) / count + 1)


def main():
    pretrain = Pretrain(dataset='dataset/raw_data_2.npy', epoch=10)
    # pretrain.net.load_state_dict(torch.load("../models/pretrained.pkl"))
    # pretrain.train()
    start_time = time.time()
    pretrain.test(load_dir="models/dnn.pkl", test_case_count=10000)
    end_time = time.time()
    print('use time: ', (end_time - start_time) / 10000)


if __name__ == '__main__':
    main()
