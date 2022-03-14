import ray
import torch
from torch.distributions import Normal

from buffer import Batch_Storage
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", UserWarning)
    from pettingzoo.sisl import multiwalker_v7
import numpy as np
from pr2_policy import Combined_Walker_Helper

@ray.remote
class Walker_Worker:
    def __init__(self, size, env_params, time_penalty, device='cpu'):
        self.size = size
        self.env_set = [multiwalker_v7.parallel_env(**env_params) for _ in range(self.size)]
        self.finished_envs = np.zeros(self.size, dtype=bool)
        self.obs_shape = (31,)
        self.agent_names = np.array(self.env_set[0].possible_agents)
        self.num_agents = len(self.agent_names)
        self.num_actions = 4
        self.reward_scale = 1
        self.time_penalty = time_penalty
        self.storage = Batch_Storage(self.size)

        self.device=device
        self.model = Combined_Walker_Helper(device)
        self.model.eval()

        self.survival_time = 0.1/env_params['max_cycles']

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
        obs = np.zeros((self.size, self.num_agents, ) + self.obs_shape)
        for env_ix in range(0, self.size):
            obs_env = self.env_set[env_ix].reset()
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
        infos = np.zeros((num_unfinished_envs, self.num_agents), bool)

        for ix in range(0, num_unfinished_envs):
            unfinished_env_ix = unfinished_env_ixs[ix]
            obs_temp, rewards_temp, dones_temp, infos_temp = \
                self.env_set[unfinished_env_ix].step(self.action_array_to_dict(actions[ix]))
            dones[ix] = all(dones_temp.values())
            self.finished_envs[unfinished_env_ix] = dones[ix]
            rewards[ix] = np.array(list(rewards_temp.values()), dtype=float)[:, np.newaxis] + self.survival_time
            for agent_ix in range(self.num_agents):
                agent = self.agent_names[agent_ix]
                next_observations[ix, agent_ix] = obs_temp[agent]
                infos[ix, agent_ix] = infos_temp[agent]

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
        while not all(self.finished_envs):
            #adj_matrix = self.get_adj_matrix() <- normal, but we gonna make it fully connected
            adj_matrix = np.tile((~np.eye((self.num_agents))[np.newaxis, ...].astype(bool)), (len(observations), 1, 1))
            #actions = self.random_action(available_actions)
            actions = self.get_action_from_model(observations, adj_matrix)
            next_observations, rewards, done_envs_cp, dones, _ = self.step(actions)
            self.storage.add_to_batch(done_envs_cp, observations, actions, rewards, next_observations, dones)
            observations = next_observations[~dones, ...]

        if not self.storage.ready_for_sending():
            return self.train()
        else:
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
            next_observations, rewards, done_envs_cp, dones, _ = self.step(actions)
            for env_ix in range(0, len(done_envs_cp)):
                if not done_envs_cp[env_ix]:
                    reward_data[env_ix].append(rewards[env_ix])
            observations = next_observations[~dones, ...]
            t += 1
        return reward_data

    def get_action_from_model(self, input_obs, adj_matrix):
        observations = [input_obs[:, agent_ix, ...] for agent_ix in range(self.num_agents)]
        encodings = [torch.tensor(obs, dtype=torch.float32).to(self.device) for obs in observations]
        output_action_dist = self.model(encodings, None)
        output = torch.cat([dist.unsqueeze(1) for dist in output_action_dist], dim=1) # batch x agents x 8
        output_means = output[:,:, 0:4]
        output_log_std = torch.full(output_means.shape, fill_value=-0.91).to(self.device)
        torch_dist = Normal(loc=output_means, scale=torch.exp(output_log_std))
        batch_torch_actions = torch_dist.sample().clamp(min=-1, max=1)
        return batch_torch_actions.cpu().detach().numpy()

    def random_action(self, available_actions):
        num_envs = len(available_actions)
        actions = np.zeros((num_envs, self.num_agents),dtype=int)
        for env_ix in range(0, num_envs):
            for agent_ix in range(0, self.num_agents):
                temp_avail_actions = np.arange(0, self.num_actions)[available_actions[env_ix, agent_ix]]
                actions[env_ix, agent_ix] = np.random.choice(temp_avail_actions, 1)
        return actions

def get_nagents_action_space():
    return 2, 4

if __name__ == '__main__':
    worker_count = 2
    ray.init(num_cpus = worker_count, num_gpus=1, _node_ip_address="0.0.0.0")
    env_params = {
        'ball_speed': 15, 'left_paddle_speed': 13, 'right_paddle_speed': 13,
        'cake_paddle': False, 'max_cycles': 500, 'bounce_randomness': False
    }
    workers = [Walker_Worker.remote(2, env_params, 0.0) for _ in range(0, worker_count)]
    output = ray.get([env.train.remote() for env in workers])
    ray.get([env.close.remote() for env in workers])
    ray.shutdown()
    for worker in range(worker_count):
        for batch in output[worker]:
            print([item.shape for item in batch])
    """ray.init(num_cpus = 5, num_gpus=1, _node_ip_address="0.0.0.0")
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    workers = [StarCraft_Worker.remote(1, {'map_name': '3m', 'reward_only_positive': False}, 0.0) for _ in range(0, 5)]
    ray.get([env.train.remote() for env in workers])
    print(ray.get([env.close.remote() for env in workers]))"""
