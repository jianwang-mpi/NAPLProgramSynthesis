import networkx as nx
import json
import env.action as action
import numpy as np

class Network:

    def to_bitvector(self, edges):
        state = np.zeros(len(self.full_edges))
        for i in range(len(self.full_edges)):
            edges = set(edges)
            if self.full_edges[i] in edges:
                state[i] = 1
        return state

    def get_edges_from_path(self, path):
        edges = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            edges.append((source, target))
        return edges

    def __init__(self, network_path):
        self.G = nx.DiGraph()
        self.full_edges = [('0', '1'), ('1', '2'), ('2', '3'), ('3', '4'), ('4', '5'), ('5', '0'), ('0', '3'), ('4', '1'), ('5', '2')]
        self.network_path = network_path
        network_json = json.load(open(self.network_path))
        for item in network_json['Nodes']:
            self.G.add_node(item['Id'], delay=item['Properties']['Delay'], hopcount=item['Properties']['HopCount'])
        for item in network_json['Links']:
            self.G.add_edge(item['Source'], item['Target'], bandwidth=item['Properties']['Bandwidth'],
                            cost=item['Properties']['Cost'], delay=item['Properties']['Delay'], )

    def delete_node(self, delete_node_action_id):
        node = action.delete_node_action[delete_node_action_id]
        self.G.remove_node(node)
        return self.to_bitvector(self.G.edges)

    def delete_edge(self, delete_edge_action_id):
        edge = action.delete_edge_action[delete_edge_action_id]
        source = edge[0]
        target = edge[1]
        self.G.remove_edge(source, target)
        return self.to_bitvector(self.G.edges)

    def filter(self, delete_path_info):
        path = action.filter_action[delete_path_info]
        edges = self.get_edges_from_path(path)
        self.G.remove_edges_from(edges)
        return self.to_bitvector(self.G.edges)

    # def find_path(self, source, target, condition):
    #     result = None
    #     for path in nx.all_simple_paths(self.G, source, target):
    #         condition = condition.split(' ')
    #         attribute = condition[0]
    #         num = condition[2]

    def find_path(self, find_path_info):
        source = action.find_path_source[find_path_info[0]]
        target = action.find_path_target[find_path_info[1]]
        p = nx.shortest_path(self.G, source, target)
        self.G.subgraph(p)
        edges = self.get_edges_from_path(p)
        return self.to_bitvector(edges)


if __name__ == '__main__':
    network = Network('default_network.json')
    edges = network.__getattribute__(action.action[1])(action.delete_edge_action[1])
    edges = network.__getattribute__(action.action[3])(action.find_path_source[0], action.find_path_target[4])
    print(edges)
