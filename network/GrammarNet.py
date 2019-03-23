import torch
from network.FullConnected import Net
import env.action as action_def
from env.env import Env

env = Env()
N_STATES = env.n_state


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
