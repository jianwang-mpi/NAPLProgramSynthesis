from env.network import Network
import numpy as np
import env.action as action


class Env:
    def __init__(self, max_len=4):
        self.network = Network('env/default_network.json')
        self.state = np.ones(len(self.network.full_edges))
        self.target = self.random_select_target()
        self.max_len = max_len
        print(self.target)

    def random_select_target(self):
        source_target = np.random.choice(action.find_path_source, size=2, replace=False)
        source_node = source_target[0]
        target_node = source_target[1]
        edges = self.network.find_path(source_node, target_node)
        return edges

    def reset(self):
        self.network = Network('env/default_network.json')
        self.state = np.ones(len(self.network.full_edges))
        self.edges = self.random_select_target()

    def step(self, root_action_id, leaf_action_info):
        root_action = action.action[root_action_id]
