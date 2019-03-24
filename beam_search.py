import torch
import torch.nn as nn
from network.GrammarNet import GrammarNet
from env.env import Env
from tqdm import tqdm
import numpy as np
import copy
import pickle
from copy import deepcopy
import sys


def topk_actions(action_value, beam_size):
    root_result = action_value[0][0]
    leaf_result_list = action_value[1]
    root_action = torch.argmax(root_result).item()
    root_value = torch.max(root_result).item()
    if root_action != 3:
        scores, indices = torch.topk(leaf_result_list[root_action][0], beam_size)
        scores = scores * root_value
        candidates = []
        for i in range(beam_size):
            candidates.append([scores[i].item(), (root_action, indices[i].item())])
    else:
        candidates = []
        source_values = leaf_result_list[3][0][:int(len(leaf_result_list[3][0]) / 2)]
        target_values = leaf_result_list[3][0][int(len(leaf_result_list[3][0]) / 2):]
        for i in range(len(source_values)):
            for j in range(len(target_values)):
                candidates.append(
                    [source_values[i].item() * target_values[j].item() * root_value, (root_action, (i, j))])
        candidates = sorted(candidates, key=lambda x: x[0], reverse=True)
        candidates = candidates[:beam_size]
    return candidates


def beam_search(net, beam_size=3):
    states = []
    probs = []
    trajectories = []
    env = Env()
    for i in range(beam_size):
        states.append(deepcopy(env))
        probs.append(1.0)
        trajectories.append([])
    for _ in range(4):
        candidate_states = []
        for k in range(beam_size):
            s = states[k].get_current_state()
            x = torch.unsqueeze(torch.FloatTensor(s), 0)
            # input only one sample
            actions_value = net(x)
            candidates = topk_actions(actions_value, beam_size)
            for i in range(beam_size):
                # step
                env = deepcopy(states[k])
                action = candidates[i][1]
                temp_traj = copy.copy(trajectories[k])
                temp_traj.append((states[k].get_current_state(), action))
                s_, r, done = env.step(action[0], action[1])
                new_state = env
                if r > 0:
                    return True, temp_traj
                # if (new_state, candidates[i][0] * probs[k], temp_traj) not in candidate_states:
                #     candidate_states.append((new_state, candidates[i][0] * probs[k], temp_traj))
                candidate_states.append((new_state, candidates[i][0] * probs[k], temp_traj))
        candidate_states = sorted(candidate_states, key=lambda x: x[1], reverse=True)

        for i in range(beam_size):
            states[i] = candidate_states[i][0]
            probs[i] = candidate_states[i][1]
            trajectories[i] = candidate_states[i][2]

    return False, None


def do_search(model_path, test_count=200, beam_size=3, save=False):
    net = GrammarNet()
    net.load_state_dict(torch.load(model_path))
    total_length = 0.0
    count = 0
    logs = []

    for i in tqdm(range(test_count)):
        result, trajectory = beam_search(net, beam_size=beam_size)
        if result:
            count += 1
            total_length += len(trajectory)
            if save:
                logs.extend(trajectory)

    if save:
        np.save('dataset/beam_search.npy', logs)

    print('success rate is: {}'.format(float(count) / test_count))
    print('average length is: {}'.format(float(total_length) / count))


if __name__ == '__main__':
    model_path = 'models/dqn.pkl'
    beam_size = 3

    do_search(model_path=model_path, beam_size=beam_size, test_count=2000, save=False)
