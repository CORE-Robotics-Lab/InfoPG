import numpy as np
from pettingzoo.sisl import multiwalker_v7
import ray

@ray.remote
class Walker_Worker:
    def __init__(self, size, env_params):
        self.size = size
        self.env_set = [multiwalker_v7.parallel_env(**env_params) for i in range(0, self.size)]
        self.finished_envs = np.zeros(self.size, dtype=np.bool)
        self.obs_shape = (31,)
        self.agent_names = ['walker_%s' % (agent_ix) for agent_ix in range(0, env_params['n_walkers'])]
        self.num_agents = len(self.agent_names)
        self.reward_scale = 1
        self.old_observations = np.zeros((self.size, self.num_agents, ) + self.obs_shape)
        self.survival_time = 0.1/env_params['max_cycles']

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
        self.finished_envs = np.zeros(self.size, dtype=np.bool)
        self.old_observations = np.zeros((self.size, self.num_agents, ) + self.obs_shape)
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

        :param actions: torch.tensor(num_left_batches x number of agents x obs_shape)
        :return: next_observations: torch.tensor(num_left_batches - *finished envs* x number of agents x obs_shape)
        :return: rewards: torch.tensor(num_left_batches x number of agents x 1)
        :return: dones: torch.tensor(num_left_batches)
        """
        unfinished_env_ixs = self.get_unfinished_env_ixs()
        num_unfinished_envs = len(unfinished_env_ixs)
        done_envs_cp = list(self.finished_envs) # this a reference to the envs we are operating on at the beginning of the step

        assert len(actions) == num_unfinished_envs, 'Error here'

        next_observations = np.zeros((num_unfinished_envs, self.num_agents, ) + self.obs_shape, dtype=np.float)
        rewards = np.zeros((num_unfinished_envs, self.num_agents, 1), np.float)
        dones = np.zeros((num_unfinished_envs), np.bool)
        infos = np.zeros((num_unfinished_envs, self.num_agents), np.bool)

        for ix in range(0, num_unfinished_envs):
            unfinished_env_ix = unfinished_env_ixs[ix]
            obs_temp, rewards_temp, dones_temp, infos_temp = \
                self.env_set[unfinished_env_ix].step(actions[ix])
            dones[ix] = all(dones_temp.values())
            self.finished_envs[unfinished_env_ix] = dones[ix]
            rewards[ix] = np.array(list(rewards_temp.values()), dtype=np.float)[:, np.newaxis] + self.survival_time
            for agent_ix in range(self.num_agents):
                agent = self.agent_names[agent_ix]
                next_observations[ix, agent_ix] = obs_temp[agent]
                infos[ix, agent_ix] = infos_temp[agent]

        next_observations = next_observations[~dones, ...]
        self.old_observations[~self.finished_envs, ...] = next_observations
        #infos = infos[~dones, ...] <- this step goes in main code, because it is needed for
        if any(dones):
            assert len(next_observations) < num_unfinished_envs, 'Next Observations has to decrease in size as environments are finished'
        return next_observations, rewards, done_envs_cp, dones, infos

    def render(self):
        return np.concatenate([self.env_set[i].render(mode='rgb_array')[np.newaxis, ...] for i in range(0, self.size)], axis=0)

if __name__ == '__main__':
    env_params = {
        'n_walkers': 2, 'position_noise': 1e-3, 'angle_noise': 1e-3, 'local_ratio': 1.0,
        'forward_reward': 1.0, 'terminate_reward':-10.0, 'fall_reward':-5.0, 'terminate_on_fall': True,
        'remove_on_fall': True, 'max_cycles': 500, 'use_package': True
    }

    env = Walker_Worker(10, env_params)
