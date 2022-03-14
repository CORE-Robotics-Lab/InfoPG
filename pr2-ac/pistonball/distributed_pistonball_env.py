import ray
import torch
from buffer import Batch_Storage
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", Warning)
    from pettingzoo.butterfly import pistonball_v4
import numpy as np
from pr2_policy import Combined_Piston_Helper
from alex_net_pistonball import Encoder
import os

@ray.remote
class Piston_Worker:
    def __init__(self, size, env_params, time_penalty, device='cpu'):
        print("ray.get_gpu_ids(): {}".format(ray.get_gpu_ids()))
        print("CUDA_VISIBLE_DEVICES: {}".format(os.environ["CUDA_VISIBLE_DEVICES"]))
        self.size = size
        self.env_set = [pistonball_v4.parallel_env(**env_params) for _ in range(self.size)]
        self.finished_envs = np.zeros(self.size, dtype=bool)
        self.obs_shape = (457, 120, 3)
        self.max_cycles = env_params['max_cycles']
        self.agent_names = np.array(self.env_set[0].possible_agents)
        self.num_agents = len(self.agent_names)
        self.num_actions = 3
        self.old_observations = np.zeros((self.size, self.num_agents, ) + self.obs_shape)
        self.reward_scale = 0.1 * ((40 * self.num_agents) - 2 * 40)
        self.early_reward_benefit = 0.5
        self.time_penalty = time_penalty
        self.storage = Batch_Storage(self.size)

        self.device=device
        self.model = Combined_Piston_Helper(device)
        self.model.eval()

        self.encoder = Encoder(device)

    def set_model_weights(self, weights):
        self.model.load_state_dicts(weights, map_location=self.device)
        self.model.eval()
        return True

    def get_model_weights(self):
        return self.model.state_dicts()

    def get_adj_matrix(self):
        """

        :return: adjacency matrix from envs that are unfinished
        """
        num_unfinished_envs = len(self.get_unfinished_env_ixs())
        adj_matrix = np.zeros((num_unfinished_envs, self.num_agents, self.num_agents), bool)
        for i in range(0, num_unfinished_envs):
            for j in range(0, self.num_agents):
                if j - 1 >= 0:
                    adj_matrix[i][j][j - 1] = True
                if j + 1 < self.num_agents:
                    adj_matrix[i][j][j + 1] = True
        return adj_matrix

    def get_unfinished_env_ixs(self):
        """
        Get Unfinished Environment Indeces

        :return: Indeces of unfinished environments
        """
        return np.argwhere(self.finished_envs == False).flatten()

    def reset(self):
        """
        Resets the environments and returns a numpy array of shape self.batch_size x number of agents x obs_shape

        :return: np.array(batch x agents x obs_shape)
        """
        self.finished_envs = np.zeros(self.size, dtype=bool)
        self.old_observations = np.zeros((self.size, self.num_agents, ) + self.obs_shape)
        obs = np.zeros((self.size, self.num_agents, ) + self.obs_shape)
        for env_ix in range(0, self.size):
            obs_env = self.env_set[env_ix].reset()
            for agent_ix in range(self.num_agents):
                obs[env_ix, agent_ix] = obs_env[self.agent_names[agent_ix]]
        return obs

    def step(self, actions, t):
        """
        Steps through the environment and returns next observations, rewards, and dones. All outputs except next_observations
        should have dim[0] == num_left_batches. The reason next_observations doesnt follow this pattern is because some
        envs might finish during this step, so len(next_observations) should decrease

        :param actions: list of dictionaries actions
        :return: next_observations: np.array(num_left_batches - *finished envs* x number of agents x obs_shape)
        :return: rewards: np.array(num_left_batches x number of agents x 1)
        :return: dones: np.array(num_left_batches)
        """
        unfinished_env_ixs = self.get_unfinished_env_ixs()
        num_unfinished_envs = len(unfinished_env_ixs)
        done_envs_cp = list(self.finished_envs) # this a reference to the envs we are operating on at the beginning of the step

        assert len(actions) == num_unfinished_envs, 'Error here'

        next_observations = np.zeros((num_unfinished_envs, self.num_agents, ) + self.obs_shape, dtype=float)
        rewards = np.zeros((num_unfinished_envs, self.num_agents, 1), float)
        dones = np.zeros((num_unfinished_envs), bool)
        infos = np.zeros((num_unfinished_envs, self.num_agents), bool)

        for ix in range(0, num_unfinished_envs):
            unfinished_env_ix = unfinished_env_ixs[ix]
            obs_temp, rewards_temp, dones_temp, infos_temp = \
                self.env_set[unfinished_env_ix].step(self.action_array_to_dict(actions[ix]))
            dones[ix] = all(dones_temp.values())
            self.finished_envs[unfinished_env_ix] = dones[ix]
            rewards[ix] = np.array(list(rewards_temp.values()), dtype=float)[:, np.newaxis]/self.reward_scale - self.time_penalty
            if dones[ix] and t < 0.5 * self.max_cycles:
                rewards[ix] += self.early_reward_benefit
            for agent_ix in range(self.num_agents):
                agent = self.agent_names[agent_ix]
                next_observations[ix, agent_ix] = obs_temp[agent]
                infos[ix, agent_ix] = infos_temp[agent]

        #next_observations = next_observations[~dones, ...]
        subtracted_observations = np.zeros((num_unfinished_envs, self.num_agents, ) + self.obs_shape, dtype=float)
        subtracted_observations[~dones, ...] = next_observations[~dones, ...]
        #infos = infos[~dones, ...] <- this step goes in main code, because it is needed for
        #if any(dones):
        #    assert len(subtracted_observations) < num_unfinished_envs, 'Next Observations has to decrease in size as environments are finished'
        return subtracted_observations, rewards, done_envs_cp, dones, infos

    def action_array_to_dict(self, actions):
        return {agent_name: actions[agent_ix] for agent_ix, agent_name in enumerate(self.agent_names)}

    def close(self):
        for env in self.env_set:
            env.close()
        return True

    def train(self, sleep_time=0):
        observations = self.reset()
        t = 0
        while not all(self.finished_envs):
            #adj_matrix = self.get_adj_matrix() <- normal, but we gonna make it fully connected
            adj_matrix = np.tile((~np.eye((self.num_agents))[np.newaxis, ...].astype(bool)), (len(observations), 1, 1))
            #actions = self.random_action(available_actions)
            actions = self.get_action_from_model(observations, adj_matrix)
            next_observations, rewards, done_envs_cp, dones, _ = self.step(actions, t)
            encodings = torch.cat([self.encoder(observations[:, agent_ix]).unsqueeze(1) for agent_ix in range(0, self.num_agents)], dim=1)
            next_encodings = torch.cat([self.encoder(next_observations[:, agent_ix]).unsqueeze(1) for agent_ix in range(0, self.num_agents)], dim=1)
            #self.storage.add_to_batch(done_envs_cp, observations, actions, rewards, next_observations, dones)
            self.storage.add_to_batch(done_envs_cp, encodings.cpu().detach().numpy(),
                                      actions, rewards, next_encodings.cpu().detach().numpy(), dones)
            observations = next_observations[~dones, ...]
            t+=1
            #print('here')
        if not self.storage.ready_for_sending():
            #print('recursiing train...')
            return self.train()
        else:
            #print('returning data!')
            return self.storage.get_data()

    def eval(self):
        observations = self.reset()
        #print(self.finished_envs)
        reward_data = [[] for _ in range(0, self.size)]
        t = 0
        while not all(self.finished_envs):
            #adj_matrix = self.get_adj_matrix() <- normal, but we gonna make it fully connected
            adj_matrix = np.tile((~np.eye((self.num_agents))[np.newaxis, ...].astype(bool)), (len(observations), 1, 1))
            #actions = self.random_action(available_actions)
            actions = self.get_action_from_model(observations, adj_matrix)
            next_observations, rewards, done_envs_cp, dones, _ = self.step(actions, t)
            encodings = torch.cat([self.encoder(observations[:, agent_ix]).unsqueeze(1) for agent_ix in range(0, self.num_agents)], dim=1)
            next_encodings = torch.cat([self.encoder(next_observations[:, agent_ix]).unsqueeze(1) for agent_ix in range(0, self.num_agents)], dim=1)
            for env_ix in range(0, len(done_envs_cp)):
                if not done_envs_cp[env_ix]:
                    reward_data[env_ix].append(rewards[env_ix])
            observations = next_observations[~dones, ...]
            t += 1
        return reward_data

    def get_action_from_model(self, input_obs, adj_matrix):
        observation_list = [input_obs[:, agent_ix] for agent_ix in range(0, self.num_agents)]
        encodings = [self.encoder(obs) for obs in observation_list]
        adj_matrix_torch = torch.tensor(adj_matrix, dtype=torch.bool).to(self.device)
        output_action_dist = self.model(encodings, adj_matrix_torch)
        reformatted_action_dist = torch.cat([dist.unsqueeze(1) for dist in output_action_dist], dim=1) # batch x agents x 3
        torch_dist = torch.distributions.Categorical(probs=reformatted_action_dist)
        batched_actions = torch_dist.sample()
        return batched_actions.cpu().detach().numpy()

    def random_action(self, available_actions):
        num_envs = len(available_actions)
        actions = np.zeros((num_envs, self.num_agents),dtype=int)
        for env_ix in range(0, num_envs):
            for agent_ix in range(0, self.num_agents):
                temp_avail_actions = np.arange(0, self.num_actions)[available_actions[env_ix, agent_ix]]
                actions[env_ix, agent_ix] = np.random.choice(temp_avail_actions, 1)
        return actions

def get_nagents_action_space():
    return 5, 3
