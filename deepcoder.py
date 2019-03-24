import time
from env.network import Network
from utils.timelimit import time_limit
from copy import deepcopy
import env.action as action
from env.env import Env
import numpy as np


class DFS:

    def __init__(self, max_depth=4):
        self.action = []
        self.max_depth = max_depth

    def reset(self):
        self.action = []

    def do_action(self, env, depth, root_action_id, leaf_action_info):
        new_state, reward, done = env.step(root_action_id, leaf_action_info)
        self.action.append((root_action_id, leaf_action_info))
        if reward >= 1:
            return True
        success = self.dfs(env, depth + 1)
        if success:
            return True
        self.action.pop(-1)
        return False

    def dfs(self, env, depth=0):
        if depth >= self.max_depth:
            return False
        temp_env = deepcopy(env)

        for i in range(len(action.action)):
            if i == 0:
                for j in range(6):
                    env = deepcopy(temp_env)
                    success = self.do_action(env, depth, i, j)
                    if success:
                        return True
            elif i == 1:
                for j in range(9):
                    env = deepcopy(temp_env)
                    success = self.do_action(env, depth, i, j)
                    if success:
                        return True
            elif i == 2:
                for j in range(2):
                    env = deepcopy(temp_env)
                    success = self.do_action(env, depth, i, j)
                    if success:
                        return True
            # elif i == 3:
            #     for j in range(6):
            #         for k in range(6):
            #             env = deepcopy(temp_env)
            #             success = self.do_action(env, depth, i, (j, k))
            #             if success:
            #                 return True
            elif i == 3:
                env = deepcopy(temp_env)
                source_node = int(env.source_node)
                target_node = int(env.target_node)
                success = self.do_action(env, depth, i, (source_node, target_node))
                if success:
                    return True
        return False


@time_limit(10)
def dfs_search(target=None):
    env = Env()
    dfs = DFS()
    initial_and_target_state = env.get_current_state()
    start_time = time.clock() * 1000
    success = dfs.dfs(env)
    print(dfs.action)
    end_time = time.clock() * 1000
    print('time: {} ms'.format(end_time - start_time))
    return success, end_time - start_time, initial_and_target_state, dfs.action

if __name__ == '__main__':
    total_time = 0.0
    total_amount = 600
    count = 0
    total_code_length = 0

    dataset = []
    for i in range(total_amount):
        try:
            success, estimated_time, initial_and_target_state, actions = dfs_search()
            if success:
                count += 1
                total_time += estimated_time
                total_code_length += len(actions)
                dataset.append((initial_and_target_state, actions))
        except Exception as e:
            print(e)

    print('success rate: ', float(count) / total_amount)
    print('avg time: ', total_time / count)
    print('avg code length: ', float(total_code_length) / count)
    np.save('dataset/raw_data_2.npy', dataset)

