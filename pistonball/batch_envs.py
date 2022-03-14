from typing import Optional, Dict, List
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", UserWarning)
    from pettingzoo.butterfly import pistonball_v4
    from pettingzoo.butterfly import cooperative_pong_v2
from policy_base import BasePolicy, Experience
from pettingzoo.sisl import multiwalker_v7
from pettingzoo.sisl import pursuit_v3
import torch.optim as optim
import numpy as np
from abc import ABC, abstractmethod
import torch
import os

class TestBatchEnvSingle:
    def __init__(self, batch_ix, total_batches, **env_params):
        self.batch_ix = batch_ix
        self.BATCH_SIZE = total_batches
        self.AGENT_NAMES = env_params['agent_names']
        self.possible_agents = self.AGENT_NAMES
        self.OBS_SHAPE = env_params['obs_shape']

    def reset(self):
        obs = {}
        for i in range(0, len(self.AGENT_NAMES)):
            obs[self.AGENT_NAMES[i]] = i*np.ones(self.OBS_SHAPE) + self.batch_ix*len(self.AGENT_NAMES)
        return obs

    def step(self, actions):
        new_obs = {}
        rewards = {}
        dones = {}
        for i in range(0, len(self.AGENT_NAMES)):
            new_obs[self.AGENT_NAMES[i]] = i*np.ones(self.OBS_SHAPE) + self.batch_ix*len(self.AGENT_NAMES)
            rewards[self.AGENT_NAMES[i]] = actions[self.AGENT_NAMES[i]] - 5000
            dones[self.AGENT_NAMES[i]] = False
        return new_obs, rewards, dones, None

class BatchEnv(ABC):
    def __init__(self, env_name: str, batch: int, env_params: Dict, path=None, seed = None):
        # additional rewards should have the time penalty and early stoppage extra credit
        self.BATCH_SIZE = batch
        self.MAX_CYCLES = env_params['max_cycles']
        if env_name == 'pistonball_v4':
            self.envs = [pistonball_v4.parallel_env(**env_params) for i in range(0, self.BATCH_SIZE)]
            if seed is not None:
                for i in range(0, self.BATCH_SIZE):
                    self.envs[i].seed(seed)
            self.N_AGENTS = env_params['n_pistons']
            self.OBS_SHAPE = (457, 120, 3)
            self.REWARD_SCALE = 0.1 * ((40 * self.N_AGENTS) - 2 * 40)
            self.ACTION_SPACE = 3
        elif env_name == 'pong':
            self.envs = [cooperative_pong_v2.parallel_env(**env_params) for i in range(0, self.BATCH_SIZE)]
            self.N_AGENTS = 2
            self.OBS_SHAPE = (280, 240, 3)
            self.ACTION_SPACE = 2
            self.REWARD_SCALE = 1
        elif env_name == 'multiwalker_v7':
            self.envs = [multiwalker_v7.parallel_env(**env_params) for i in range(0, self.BATCH_SIZE)]
            self.N_AGENTS = env_params['n_walkers']
            self.OBS_SHAPE = (31,)
            self.REWARD_SCALE = 1
            self.ACTION_SPACE = 4
        elif env_name == 'pursuit_v3':
            self.envs = [pursuit_v3.parallel_env(**env_params) for i in range(0, self.BATCH_SIZE)]
            self.N_AGENTS = env_params['n_pursuers']
            self.OBS_RANGE = env_params['obs_range']
            self.OBS_SHAPE = (self.OBS_RANGE, self.OBS_RANGE, 3)
            self.ACTION_SPACE = 5
            self.REWARD_SCALE = 1
        elif env_name == 'test':
            self.envs = [TestBatchEnvSingle(i, self.BATCH_SIZE, **env_params) for i in range(0, self.BATCH_SIZE)]
            self.N_AGENTS = len(env_params['agent_names'])
            self.OBS_SHAPE=env_params['obs_shape']
        else:
            raise Exception("%s isn't supported yet!" % (env_name))
        self.AGENT_NAMES = np.array(self.envs[0].possible_agents)
        self.adj_matrix = np.zeros((self.BATCH_SIZE, len(self.AGENT_NAMES), len(self.AGENT_NAMES)))
        for i in range(0, self.BATCH_SIZE):
            for j in range(0, len(self.AGENT_NAMES)):
                if j - 1 >= 0:
                    self.adj_matrix[i][j][j - 1] = 1
                if j + 1 < len(self.AGENT_NAMES):
                    self.adj_matrix[i][j][j + 1] = 1
        self.update_adj_matrix()
        if path is not None:
            self.path = path
            self.save = True
            print('**Saving policies and optimizers to: %s' % self.path)
        else:
            self.save = False

        self.DONE_ENVS = np.array([False for _ in range(0, self.BATCH_SIZE)], dtype=np.bool)

    def get_agent_names(self):
        return self.AGENT_NAMES

    def batch_reset(self):
        self.DONE_ENVS = np.array([False for _ in range(0, self.BATCH_SIZE)])
        ret_obs = {agent_name: np.zeros((self.BATCH_SIZE,) + self.OBS_SHAPE) for agent_name in self.AGENT_NAMES}
        for batch_ix in range(0, self.BATCH_SIZE):
            obs = self.envs[batch_ix].reset()
            for agent_name, agent_obs in obs.items():
                ret_obs[agent_name][batch_ix] = agent_obs
        return ret_obs

    def get_batch_obs_shapes(self, obs):
        shape = {k: obs[k].shape for k in obs.keys()}
        return shape

    @abstractmethod
    def loop(self, user_params: Dict, policies: Dict[str, BasePolicy], optimizers: Dict[str, optim.Optimizer], schedulers):
        pass

    def update_adj_matrix(self):
        # this is environment dependent, and on different envs, the adjacency matrix should change
        pass

    def batch_step(self, actions: Dict[str, List], step_num: int, time_penalty: float, early_reward_benefit: float):
        num_left_batches = np.count_nonzero(self.DONE_ENVS == False)
        left_batches = np.where(self.DONE_ENVS == False)[0]

        next_observations = {agent: np.zeros((num_left_batches,) + self.OBS_SHAPE) for agent in self.AGENT_NAMES}
        rewards = {agent: np.zeros((num_left_batches)) for agent in self.AGENT_NAMES}
        dones = {agent: np.zeros((num_left_batches), dtype=np.bool) for agent in self.AGENT_NAMES}
        for batch_ix in range(0, num_left_batches):
            actual_batch_ix = left_batches[batch_ix]
            obs_temp, rewards_temp, dones_temp, _ = self.envs[actual_batch_ix].step({agent: batched_action[batch_ix] for agent, batched_action in actions.items()})
            batch_finished = all(dones_temp.values())
            if batch_finished:
                self.DONE_ENVS[actual_batch_ix] = True
            for agent in self.AGENT_NAMES:
                reward = (rewards_temp[agent] / self.REWARD_SCALE) - abs(time_penalty)
                if self.DONE_ENVS[actual_batch_ix] and step_num < 0.5 * self.MAX_CYCLES:
                    reward += early_reward_benefit
                next_observations[agent][batch_ix] = obs_temp[agent]
                rewards[agent][batch_ix] = reward
                dones[agent][batch_ix] = dones_temp[agent]
        return next_observations, rewards, dones

    def consensus_update(self, policies):
        vnet_copies = {agent: policies[agent].policy.v_net.state_dict() for agent in policies.keys()}
        for agent_ix in range(0, self.N_AGENTS):
            neighbor_name_set = set()
            for batch_ix in range(0, self.BATCH_SIZE):
                neighbor_names = self.AGENT_NAMES[self.adj_matrix[batch_ix, agent_ix] == 1]
                for neighbor in neighbor_names:
                    if neighbor not in neighbor_name_set:
                        neighbor_name_set.add(neighbor)
            if len(neighbor_name_set) == 0:
                continue
            neighbor_vnet_copies = [vnet_copies[name] for name in neighbor_name_set]
            policies[self.AGENT_NAMES[agent_ix]].consensus_update(neighbor_vnet_copies)

    def initialize_memory(self):
        memory = {}  # this is a map of agent_name -> a list of length self.BATCH_SIZE, but batches that are done are set to None
        for agent in self.AGENT_NAMES:
            memory[agent] = []
            for actual_batch_ix in range(0, self.BATCH_SIZE):
                if not self.DONE_ENVS[actual_batch_ix]:
                    memory[agent].append(Experience())
                else:
                    memory[agent].append(None)
            assert len(memory[agent]) == self.BATCH_SIZE, 'Error Here'
        return memory

    def k_level_communication(self, policies, policy_initial, num_left_batches, left_batches, k_levels):
        for k in range(0, k_levels):
            output_dist = {}
            for agent_ix, agent in enumerate(self.AGENT_NAMES):
                batched_neighbors = [[] for _ in range(0, num_left_batches)] # for each batch, the policies of agent
                for batch_ix in range(0, num_left_batches):
                    actual_batch_number = left_batches[batch_ix]
                    neighbor_mask_for_agent = (self.adj_matrix[actual_batch_number][agent_ix] == 1)
                    neighbor_names = self.AGENT_NAMES[neighbor_mask_for_agent]
                    for neighbor in neighbor_names:
                        batched_neighbors[batch_ix].append([policy_initial[neighbor][batch_ix], neighbor, batch_ix])
                latent_vector = policies[agent].forward(policy_initial[agent], 1, batched_neighbors)
                output_dist[agent] = latent_vector
            policy_initial = output_dist
        return policy_initial

    def add_rewards_to_memory(self, policies, memory, rewards, num_left_batches, left_batches):
        for agent in self.AGENT_NAMES:
            for batch_ix in range(0, num_left_batches):
                actual_batch_number = left_batches[batch_ix]
                memory[agent][actual_batch_number].rewards = rewards[agent][batch_ix]
            policies[agent].add_to_memory(memory[agent])

    def conclude_step(self, next_observations, dones):
        observations = {}
        for agent in self.AGENT_NAMES:
            observations[agent] = next_observations[agent][~dones[agent]]
        self.update_adj_matrix()
        return observations

    def compute_epoch_data(self, policies, verbose=True, eval=False, standardize_rewards=False):
        epoch_data = {}
        iterations = 0
        for agent in self.AGENT_NAMES:
            loss, batched_mean_reward, batched_length_iteration = policies[agent].compute_loss(standardize_rewards=standardize_rewards)
            mean_reward = batched_mean_reward.mean()
            mean_iteration_length = batched_length_iteration.mean()
            iterations += mean_iteration_length
            epoch_data[agent] = [mean_reward, mean_iteration_length, batched_mean_reward, batched_length_iteration]
            #print('Performing backprop on %s' % (agent))
            if verbose:
                print('\t Reward for %s: %f' % (agent, mean_reward))
            if not eval:
                loss.backward(retain_graph=True)
        return epoch_data, iterations/self.N_AGENTS

    def conclude_epoch(self, policies, optimizers, schedulers):
        for agent in self.AGENT_NAMES:
            optimizers[agent].step()
            if schedulers is not None:
                schedulers[agent].step()

        self.clear_memory(policies)

    def clear_memory(self, policies):
        for agent in self.AGENT_NAMES:
            policies[agent].clear_memory()

    def render(self):
        assert self.BATCH_SIZE == 1, 'Cant render multiple batch envs'
        return self.envs[0].render(mode='rgb_array')

    def close(self):
        for i in range(0, self.BATCH_SIZE):
            self.envs[i].close()

    def save_checkpoint(self, policies, optimizers):
        if self.save:
            for agent in self.AGENT_NAMES:
                torch.save({
                    'policy': policies[agent].state_dict(),
                    'optimizer': optimizers[agent].state_dict()
                }, os.path.join(self.path, agent+'.pt'))