import numpy as np
from env.env import Env

def generate_dataset(raw_data_file_path):
    raw_data = np.load(raw_data_file_path)
    dataset = []
    for item in raw_data:
        state = item[0]
        init_target = state[int(len(state) / 2):]
        actions = item[1]
        env = Env(target_state=init_target)
        for action in actions:
            dataset.append((state, action))
            state, reward, done = env.step(action[0], action[1])
            if done:
                break
    np.save('dataset/dataset.npy', dataset)


if __name__ == '__main__':
    generate_dataset('dataset/raw_data.npy')