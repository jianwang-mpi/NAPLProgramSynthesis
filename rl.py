import torch

from torch.utils.data.dataloader import DataLoader
from env.bitvector import BitVector
from network.FullConnected import Net

from tqdm import tqdm
from tensorboard_logger import configure, log_value
import time
import os
from data.BitVectorDataset import BitVectorDataset
from beam_search import beam_search

env = BitVector()
N_ACTIONS = env.n_actions
N_STATES = env.max_len * 2

class Pretrain:
    def __init__(self, pretrain_epoch = 20, epoch=2000):
        self.net = Net(n_state=N_STATES, n_action=N_ACTIONS, hidden_num=1024)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net = self.net.to(self.device)
        self.dataset = BitVectorDataset('dataset_bak/bitvector_8bit_total.pkl', length_reward=True)

        self.dataloader = DataLoader(dataset=self.dataset, batch_size=16, shuffle=True, drop_last=True)
        self.pretrain_epoch = pretrain_epoch
        self.epoch = epoch

        self.loss = torch.nn.MSELoss()

        self.optimizer = torch.optim.Adam(params=self.net.parameters())

        self.test_dataloader = DataLoader(dataset=self.dataset, batch_size=1)
        self.max_acc = 0

        log_dir = 'logs/{}'.format(time.time())
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        configure(logdir=log_dir)

    def train(self):
        self.pretrain()


    def pretrain(self):
        self.net.train()
        for _ in range(self.pretrain_epoch):
            loss_val = 0
            for x, y in tqdm(iter(self.dataloader)):
                if torch.cuda.is_available():
                    x = x.to(self.device)
                    y = y.to(self.device)
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
                    x = x.to(self.device)
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
            torch.save(self.net.state_dict(), 'models/rl.pkl')
            self.max_acc = acc
        print(acc)


def main():
    pretrain = Pretrain(100)
    # pretrain.net.load_state_dict(torch.load("../models/pretrained.pkl"))
    pretrain.train()
    # pretrain.test(init_max=1, load_dir="models/pretrained_total.pkl")
    pretrain.test()

if __name__ == '__main__':
    main()
