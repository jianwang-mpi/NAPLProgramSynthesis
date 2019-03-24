import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from env.env import Env
import env.action as action_def
from network.GrammarNet import GrammarNet

# Hyper Parameters
BATCH_SIZE = 1
LR = 0.003  # learning rate
EPSILON = 0.90  # greedy policy
GAMMA = 1  # reward discount
TARGET_REPLACE_ITER = 100  # target update frequency
MEMORY_CAPACITY = 5000


class DQN(object):
    def __init__(self):
        self.eval_net, self.target_net = GrammarNet(), GrammarNet()

        self.learn_step_counter = 0  # for target updating
        self.memory_counter = 0  # for storing memory
        # self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory
        self.memory = [[]] * MEMORY_CAPACITY
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

        self.max_acc: float = 0.0

    def choose_action(self, x):
        x = torch.unsqueeze(torch.Tensor(x), 0)
        # input only one sample
        if np.random.uniform() < EPSILON:  # greedy
            root_result, leaf_result = self.eval_net(x)
            root_action = torch.argmax(root_result).item()
            if root_action != 3:
                leaf_action = torch.argmax(leaf_result[root_action]).item()
            else:
                find_path_result = leaf_result[3]
                find_path_source = torch.argmax(find_path_result[:, : int(find_path_result.shape[1] / 2)]).item()
                find_path_target = torch.argmax(find_path_result[:, int(find_path_result.shape[1] / 2):]).item()
                leaf_action = (find_path_source, find_path_target)
        else:  # random
            root_action, leaf_action = action_def.get_random_action()

        return root_action, leaf_action

    def store_transition(self, s, a, r, s_):
        transition = (s, a, r, s_)
        # replace the old memory with new memory
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index] = transition
        self.memory_counter += 1

    def learn(self):
        # target parameter update
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # sample a transition
        sample = np.random.choice(self.memory)
        b_s, b_a, b_r, b_s_ = sample
        b_s = torch.unsqueeze(torch.Tensor(b_s), 0)
        b_r = torch.unsqueeze(torch.Tensor(b_r), 0)
        b_s_ = torch.unsqueeze(torch.Tensor(b_s_), 0)

        root_action_eval, leaf_action_eval = self.eval_net(b_s)
        root_action_next, leaf_action_next = self.target_net(b_s_).detach()

        

        # q_eval w.r.t the action in experience
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)
        q_next = self.target_net(b_s_).detach()  # detach from graph, don't backpropagate
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save(self.target_net.state_dict(), path)

    def load(self, path):
        self.target_net.load_state_dict(torch.load(path))
        self.eval_net.load_state_dict(torch.load(path))

    def test(self, test_case_count=200, load_dir=None):
        self.target_net = self.target_net.eval()
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
                root_result, leaf_result = self.target_net(x)
                root_action = torch.argmax(root_result).item()
                if root_action != 3:
                    leaf_action = torch.argmax(leaf_result[root_action]).item()
                    # step
                    s_, r, done = env.step(root_action, leaf_action)
                else:
                    find_path_result = leaf_result[3]
                    find_path_source = torch.argmax(find_path_result[:, : int(find_path_result.shape[1] / 2)]).item()
                    find_path_target = torch.argmax(find_path_result[:, int(find_path_result.shape[1] / 2):]).item()
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
            torch.save(self.target_net.state_dict(), 'models/dqn.pkl')
            self.max_acc = acc
        print("acc is: ", acc)
        if count > 0:
            # 因为统计的时候少1，这里补上1
            print("length is: ", float(total_length) / count + 1)


dqn = DQN()
env = Env()
# dqn.load("models/pretrained.pkl")
print('\nCollecting experience...')
for i_episode in range(600000):
    s = env.reset()
    ep_r = 0
    for _count in range(8):
        root_action, leaf_action = dqn.choose_action(s)

        # take action
        s_, r, done = env.step(root_action, leaf_action)

        dqn.store_transition(s, (root_action, leaf_action), r, s_)

        ep_r += r
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()

        if done:
            break
        s = s_

    if i_episode % 10000 == 1:
        dqn.test()

dqn.save('models/dqn_final_no_pretrain.pkl')
