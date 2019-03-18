import networkx as nx
import json
import matplotlib.pyplot as plt
class Network:
    def __init__(self, network_path):
        self.G = nx.DiGraph()

        self.network_path = network_path
        network_json = json.load(open(self.network_path))
        for item in network_json['Nodes']:
            self.G.add_node(item['Id'], delay=item['Properties']['Delay'], hopcount=item['Properties']['HopCount'])
        for item in network_json['Links']:
            self.G.add_edge(item['Source'], item['Target'], bandwidth=item['Properties']['Bandwidth'],
                            cost=item['Properties']['Cost'], delay=item['Properties']['Delay'],)
        p = nx.shortest_path(self.G, "0", "5", "Delay")
        print(p[0])

        sub_g = nx.subgraph(self.G, p)

        nx.draw(sub_g, with_labels=True, )
        plt.show()


if __name__ == '__main__':
    network = Network('default_network.json')