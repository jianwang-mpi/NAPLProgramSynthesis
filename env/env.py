from env.network import Network
import numpy as np
import env.action as action
import networkx as nx
import copy
import random


class Env:
    def __init__(self, target_state=None, max_len=4):
        self.network = Network('env/default_network.json')
        self.backup_network = copy.deepcopy(self.network)
        self.state = np.ones(len(self.network.full_edges))
        if target_state is None:
            self.target_state = self.random_select_target()
        else:
            self.target_state = target_state
        self.max_len = max_len
        self.count = 0

        self.n_state = self.state.size * 2

    def random_select_target(self):
        source_target = np.random.choice(action.find_path_source, size=2, replace=False)
        self.source_node = source_target[0]
        self.target_node = source_target[1]
        paths = nx.all_simple_paths(self.network.G, self.source_node, self.target_node)
        path_list = []
        for path in paths:
            path_list.append(path)
        path = random.choice(path_list)
        # p = nx.shortest_path(self.network.G, source_node, target_node)
        edges = self.network.get_edges_from_path(path)
        return self.network.to_bitvector(edges)

    def reset(self, target_state=None):
        self.network = copy.deepcopy(self.backup_network)
        self.state = np.ones(len(self.network.full_edges))
        if target_state is None:
            self.target_state = self.random_select_target()
        else:
            self.target_state = target_state
        # self.target_state = [0., 0., 1., 0., 0., 0., 0., 0., 0.]
        return np.concatenate([self.state, self.target_state])

    def step(self, root_action_id, leaf_action_info):
        self.count += 1
        root_action = action.action[root_action_id]
        try:
            self.state = self.network.__getattribute__(root_action)(leaf_action_info)
        except Exception as e:
            pass
        done = self.count >= self.max_len or np.all(self.state == self.target_state)
        reward = int(np.all(self.state == self.target_state))
        return np.concatenate([self.state, self.target_state]), reward, done

    def get_current_state(self):
        return np.concatenate([self.state, self.target_state])
