from env.env import Env
from env.network import Network
import env.action as action
import numpy as np
# if __name__ == '__main__':
#     network = Network('env/default_network.json')
#     edges = network.__getattribute__(action.action[1])(action.delete_edge_action[6])
#     edges = network.__getattribute__(action.action[3])(action.find_path_source[0], action.find_path_target[4])
#     print(edges)

# if __name__ == '__main__':
#     env = Env()
#     state = env.reset()
#     print(state)
#     print(env.target_state)
#     state = env.step(3, (2, 3))
#     print(state)
#     state = env.reset()
#     print(env.network.G.edges)

if __name__ == '__main__':
    data = np.load('dataset/data.npy')
    print(data[0][0])