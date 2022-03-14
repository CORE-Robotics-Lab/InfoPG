import os
import numpy as np
from torch.distributions import Normal

import wrapped_starcraft_env as sc_env
import distributed_starcraft_policy as sc
import moa.distributed_starcraft_policy_moa as model_other

import torch
import ray

@ray.remote
def eval(name_env, model_dir, k, moa=False):
    device=torch.device("cpu")
    arr_actions = False
    if name_env == 'starcraft':
        env_params = {
            'map_name': '3m', 'reward_only_positive': True
        }
        env = sc_env.StarCraft2Env_MAF(**env_params)
        num_agents = 3
        if not moa:
            model = sc.Combined_StarCraft_Helper(device, '3m', k)
            with open(os.path.join('experiments', 'starcraft', model_dir, 'combined_model.pt'), 'rb') as f:
                d = torch.load(f, map_location=device)
        else:
            model = model_other.Combined_StarCraft_Helper(device, '3m')
            with open(os.path.join('moa', 'experiments', 'starcraft', model_dir, 'combined_model.pt'), 'rb') as f:
                d = torch.load(f, map_location=device)
        model.load_state_dicts(d['policy'])
        categorical = True
        reward_scale = 1.0
        time_penalty = 0.0
        encoder = lambda data: torch.tensor(data, dtype=torch.float32).to(device)
        arr_actions = True
        max_cycles = 100
        available_actions = True
    else:
        raise Exception(f"Dont recognize this environment: {name_env}")

    if not arr_actions:
        obs = dict_to_array_obs(env.reset())
    else:
        obs, _ = env.reset()
        obs = np.array(obs)[np.newaxis, ...]

    reward_list = []
    gamma = 0.99
    steps = 0
    while True and steps < 1000:
        if available_actions is not None:
            available_actions = env.get_avail_actions()
            available_actions = np.array(available_actions)[np.newaxis, ...]

        if categorical:
            np_actions = get_action_from_model_categorical(model, encoder, num_agents, obs, None, available_actions)
        else:
            np_actions = get_action_from_model_continuous(model, num_agents, obs)

        if not arr_actions:
            actions = {agent_name: np_actions[0, agent_ix] for agent_ix, agent_name in enumerate(env.possible_agents)}
            next_obs, rewards, dones, _  = env.step(actions)
            reward_list.append( (dict_to_array_rewards(rewards)/reward_scale) - time_penalty)
            if all(dones.values()):
                break
            obs = dict_to_array_obs(next_obs)
        else:
            actions = np_actions[0]
            rewards, dones, _ = env.step(actions)
            next_obs = env.get_obs()
            reward_list.append( (rewards[np.newaxis, ...]/reward_scale) - time_penalty)
            if dones:
                break
            obs = np.array(next_obs)[np.newaxis, ...]
        steps += 1

    env.close()

    temp = np.zeros((1, num_agents))
    cum_discounted_rewards = np.zeros((len(reward_list), num_agents))
    for time_ix in range(len(reward_list)-1, -1, -1):
        temp += gamma*reward_list[time_ix]
        cum_discounted_rewards[time_ix] = temp
    return np.array([cum_discounted_rewards.mean(axis=0).sum(axis=0), len(reward_list)])

def dict_to_array_rewards(dictionary):
    agent_names = list(dictionary.keys())
    agent_names.sort()
    return np.array(list(dictionary.values()))[np.newaxis, ...]

def dict_to_array_obs(dictionary):
    agent_names = list(dictionary.keys())
    agent_names.sort()
    return np.concatenate([dictionary[agent][np.newaxis, ...] for agent in agent_names], axis=0)[np.newaxis, ...]

def get_action_from_model_categorical(model, encoder, num_agents, input_obs, adj_matrix, available_actions):
    device = 'cpu'
    if available_actions is not None:
        available_actions_list = [torch.tensor(available_actions[:, agent_ix], dtype=torch.bool).to(device) for agent_ix in range(0, num_agents)]
    if adj_matrix is not None:
        adj_matrix_torch = torch.tensor(adj_matrix, dtype=torch.bool).to(device)
    else:
        adj_matrix_torch = None
    observation_list = [input_obs[:, agent_ix] for agent_ix in range(0, num_agents)]
    encodings = [encoder(obs) for obs in observation_list]
    if available_actions is not None:
        output_action_dist = model(encodings, adj_matrix_torch, available_actions_list)
        output_action_dist = output_action_dist[0]
    else:
        output_action_dist = model(encodings, adj_matrix_torch)
        output_action_dist = output_action_dist[0]
    reformatted_action_dist = torch.cat([dist.unsqueeze(1) for dist in output_action_dist], dim=1) # batch x agents x 3
    torch_dist = torch.distributions.Categorical(probs=reformatted_action_dist)
    batched_actions = torch_dist.sample()
    return batched_actions.cpu().detach().numpy()

def get_action_from_model_continuous(model, num_agents, input_obs):
    device = 'cpu'
    observations = [input_obs[:, agent_ix, ...] for agent_ix in range(num_agents)]
    encodings = [torch.tensor(obs, dtype=torch.float32).to(device) for obs in observations]
    output_action_dist = model(encodings, None)
    output = torch.cat([dist.unsqueeze(1) for dist in output_action_dist], dim=1) # batch x agents x 8
    output_means = output[:,:, 0:4]
    output_log_std = torch.full(output_means.shape, fill_value=-0.91).to(device)
    torch_dist = Normal(loc=output_means, scale=torch.exp(output_log_std))
    batch_torch_actions = torch_dist.sample().clamp(min=-1, max=1)
    return batch_torch_actions.cpu().detach().numpy()

def get_reward_mean(arr):
    return arr[:, 0].mean()

def get_reward_standard_error(arr):
    num_samples = len(arr)
    return arr[:, 0].std()/num_samples

def get_steps_mean(arr):
    return arr[:, 1].mean()

def get_steps_standard_error(arr):
    num_samples = len(arr)
    return arr[:, 1].std()/num_samples

if __name__ == '__main__':
    num_cpus = 50
    ray.init(num_cpus = num_cpus, _node_ip_address="0.0.0.0")
    model_map = {
        #'2021-07-14 05_18_52': 'adv',
        #'2021-08-08 17_25_10': 'infopg',
        #'2021-08-09 01_24_23': 'consensus',
        #'2021-08-08 21_03_30': 'nc_a2c',
        '2021-08-09 01_22_39': 'moa'
    }

    for model, name in model_map.items():
        k = 0
        if name == 'adv' or name == 'infopg':
            k = 1
        data = np.array(ray.get([eval.remote('starcraft', model, k, name == 'moa') for trial in range(0, 100)]))
        print(f'Starcraft for {name}:')
        print('\t', 'Mean Avg rewards: %.3f, Standard Error: %.3f' % (get_reward_mean(data), get_reward_standard_error(data)))
        print('\t', 'Mean Avg steps: %.3f, Standard Error: %.3f' % (get_steps_mean(data), get_steps_standard_error(data)))