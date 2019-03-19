import networkx as nx
import json
import matplotlib.pyplot as plt
import env.action as action

class Network:

    def get_edges_from_path(self, path):
        edges = []
        for i in range(len(path) - 1):
            source = path[i]
            target = path[i + 1]
            edges.append((source, target))
        return edges

    def __init__(self, network_path):
        self.G = nx.DiGraph()

        self.network_path = network_path
        network_json = json.load(open(self.network_path))
        for item in network_json['Nodes']:
            self.G.add_node(item['Id'], delay=item['Properties']['Delay'], hopcount=item['Properties']['HopCount'])
        for item in network_json['Links']:
            self.G.add_edge(item['Source'], item['Target'], bandwidth=item['Properties']['Bandwidth'],
                            cost=item['Properties']['Cost'], delay=item['Properties']['Delay'], )

    def delete_node(self, node):
        self.G.remove_node(node)
        return self.G.edges

    def delete_edge(self, edge):
        source = edge[0]
        target = edge[1]
        self.G.remove_edge(source, target)
        return self.G.edges

    def filter(self, path):
        edges = self.get_edges_from_path(path)
        self.G.remove_edges_from(edges)
        return self.G.edges

    # def find_path(self, source, target, condition):
    #     result = None
    #     for path in nx.all_simple_paths(self.G, source, target):
    #         condition = condition.split(' ')
    #         attribute = condition[0]
    #         num = condition[2]

    def find_path(self, source, target):
        p = nx.shortest_path(self.G, source, target)
        edges = self.get_edges_from_path(p)
        return edges


if __name__ == '__main__':
    network = Network('default_network.json')
    edges = network.__getattribute__(action.action[1])(action.delete_edge_action[1])
    edges = network.__getattribute__(action.action[3])(action.find_path_source[0], action.find_path_target[4])
    print(edges)
