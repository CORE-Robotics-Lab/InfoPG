import ray
import torch
from buffer import Batch_Storage
from wrapped_starcraft_env import StarCraft2Env_MAF
import numpy as np
from pr2_policy import Combined_Starcraft_Helper
import os
import time

@ray.remote
class StarCraft_Worker:
    def __init__(self, size, env_params, time_penalty, device='cpu'):
        self.size = size
        self.env_set = [StarCraft2Env_MAF(**env_params) for _ in range(self.size)]
        self.finished_envs = np.zeros(self.size, dtype=bool)
        self.obs_shape = (self.env_set[0].get_obs_size(), )
        self.agent_names = np.arange(0, self.env_set[0].n_agents)
        self.num_agents = len(self.agent_names)
        self.num_actions = self.env_set[0].n_actions
        self.reward_scale = 1
        self.time_penalty = time_penalty
        self.storage = Batch_Storage(self.size)

        self.device=device
        self.model = Combined_Starcraft_Helper(device=device)
        self.model.eval()
        self.move_probability = 0.1

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
        unfinished_env_ixs = self.get_unfinished_env_ixs()
        num_unfinished_envs = len(unfinished_env_ixs)
        adj_matrix = np.zeros((num_unfinished_envs, self.num_agents, self.num_agents), bool)
        for i in range(0, num_unfinished_envs):
            adj_matrix[i] = self.env_set[unfinished_env_ixs[i]].get_visibility_matrix()[:, 0:self.num_agents]
        return adj_matrix

    def curriculum_step(self):
        self.move_probability = np.minimum(1, 1.2*self.move_probability)
        return True

    def get_avail_actions(self):
        unfinished_env_ixs = self.get_unfinished_env_ixs()
        num_unfinished_envs = len(unfinished_env_ixs)
        avail_actions = np.zeros((num_unfinished_envs, self.num_agents, self.num_actions), bool)
        for i in range(0, num_unfinished_envs):
            avail_actions[i] = np.array(self.env_set[unfinished_env_ixs[i]].get_avail_actions()).astype(bool)
        choice = np.random.choice(2, 1, p=[self.move_probability, 1 - self.move_probability]).item()
        if choice == 1:
            avail_actions[:,:, [2, 3, 5]] = False
        # ix: 0 (no-action agent is dead) 1: stop 2: North 3: South 4: East 5: West 6: attack a target
        return avail_actions

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
        obs = np.zeros((self.size, self.num_agents, ) + self.obs_shape)
        for env_ix in range(0, self.size):
            obs_env, _ = self.env_set[env_ix].reset()
            for agent_ix in range(self.num_agents):
                obs[env_ix, agent_ix] = obs_env[self.agent_names[agent_ix]]
        return obs

    def step(self, actions):
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
        infos = [None for _ in range(0, num_unfinished_envs)]

        for ix in range(0, num_unfinished_envs):
            unfinished_env_ix = unfinished_env_ixs[ix]
            rewards_temp, terminated, infos_temp = \
                self.env_set[unfinished_env_ix].step(actions[ix])
            dones[ix] = terminated
            obs_temp = self.env_set[unfinished_env_ix].get_obs()
            self.finished_envs[unfinished_env_ix] = dones[ix]
            try:
                rewards[ix] = rewards_temp.astype(float)[:, np.newaxis] - self.time_penalty
            except Exception as e:
                print(rewards, rewards_temp)
                print(str(e))
                rewards[ix] = rewards_temp[:, np.newaxis] - self.time_penalty
                raise e
            for agent_ix in range(self.num_agents):
                agent = self.agent_names[agent_ix]
                next_observations[ix, agent_ix] = obs_temp[agent]
            infos[ix] = infos_temp

        subtracted_observations = np.zeros((num_unfinished_envs, self.num_agents, ) + self.obs_shape, dtype=float)
        subtracted_observations[~dones, ...] = next_observations[~dones, ...]
        #infos = infos[~dones, ...] <- this step goes in main code, because it is needed for
        #if any(dones):
        #    assert len(subtracted_observations) < num_unfinished_envs, 'Next Observations has to decrease in size as environments are finished'
        return subtracted_observations, rewards, done_envs_cp, dones, infos

    def close(self):
        for env in self.env_set:
            env.close()
        return True

    def train(self, sleep_time=0):
        observations = self.reset()
        while not all(self.finished_envs):
            available_actions = self.get_avail_actions()
            adj_matrix = self.get_adj_matrix()
            actions = self.get_action_from_model(observations, available_actions, adj_matrix)
            next_observations, rewards, done_envs_cp, dones, _ = self.step(actions)
            self.storage.add_to_batch(done_envs_cp, observations, actions, rewards, next_observations, dones)
            observations = next_observations[~dones, ...]

        if not self.storage.ready_for_sending():
            #print('Finished one epoch, but have to redo!')
            return self.train()
        else:
            return self.storage.get_data()

    def eval(self):
        observations = self.reset()
        #print(self.finished_envs)
        reward_data = [[] for _ in range(0, self.size)]
        t = 0
        while not all(self.finished_envs):
            available_actions = self.get_avail_actions()
            adj_matrix = np.tile((~np.eye((self.num_agents))[np.newaxis, ...].astype(bool)), (len(observations), 1, 1))
            #actions = self.random_action(available_actions)
            actions = self.get_action_from_model(observations, available_actions, adj_matrix)
            next_observations, rewards, done_envs_cp, dones, _ = self.step(actions)
            for env_ix in range(0, len(done_envs_cp)):
                if not done_envs_cp[env_ix]:
                    reward_data[env_ix].append(rewards[env_ix])
            observations = next_observations[~dones, ...]
            t += 1
        return reward_data

    def get_action_from_model(self, input_obs, available_actions, adj_matrix):
        observation_list = [input_obs[:, agent_ix] for agent_ix in range(0, self.num_agents)]
        available_actions_list = [torch.tensor(available_actions[:, agent_ix], dtype=torch.bool).to(self.device) for agent_ix in range(0, self.num_agents)]
        encodings = [torch.tensor(obs, dtype=torch.float32).to(self.device) for obs in observation_list]
        adj_matrix_torch = torch.tensor(adj_matrix, dtype=torch.bool).to(self.device)
        output_action_dist = self.model(encodings, available_actions_list, adj_matrix_torch)
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

def get_nagents_action_space(env_params):
    env = StarCraft2Env_MAF(**env_params)
    return env.n_agents, env.n_actions

if __name__ == '__main__':
    ray.init(num_cpus = 1, num_gpus=1, _node_ip_address="0.0.0.0")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    workers = [StarCraft_Worker.remote(1, {'map_name': '2c_vs_64zg', 'reward_only_positive': False}, 0.0) for _ in range(0, 1)]
    ray.get([env.train.remote() for env in workers])
    ray.get([env.close.remote() for env in workers])
    ray.shutdown()
    """ray.init(num_cpus = 5, num_gpus=1, _node_ip_address="0.0.0.0")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    workers = [StarCraft_Worker.remote(1, {'map_name': '3m', 'reward_only_positive': False}, 0.0) for _ in range(0, 5)]
    ray.get([env.train.remote() for env in workers])
    print(ray.get([env.close.remote() for env in workers]))"""
