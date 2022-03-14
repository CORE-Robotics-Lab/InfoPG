import ray
import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always", UserWarning)
    from pettingzoo.butterfly import cooperative_pong_v2
import numpy as np

@ray.remote
class Pong_Worker:
    def __init__(self, size, env_params, time_penalty):
        self.size = size
        self.env_set = [cooperative_pong_v2.parallel_env(**env_params) for _ in range(self.size)]
        self.finished_envs = np.zeros(self.size, dtype=np.bool)
        self.obs_shape = (280, 240, 3)
        self.agent_names = np.array(self.env_set[0].possible_agents)
        self.num_agents = len(self.agent_names)
        self.reward_scale = 1
        self.time_penalty = time_penalty
        self.old_observations = np.zeros((self.size, self.num_agents, ) + self.obs_shape)

    def get_adj_matrix(self):
        """

        :return: adjacency matrix from envs that are unfinished
        """
        num_unfinished_envs = len(self.get_unfinished_env_ixs())
        adj_matrix = np.zeros((num_unfinished_envs, self.num_agents, self.num_agents), np.bool)
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

        :param actions: list of dictionaries actions
        :return: next_observations: np.array(num_left_batches - *finished envs* x number of agents x obs_shape)
        :return: rewards: np.array(num_left_batches x number of agents x 1)
        :return: dones: np.array(num_left_batches)
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
            rewards[ix] = np.array(list(rewards_temp.values()), dtype=np.float)[:, np.newaxis] - self.time_penalty
            for agent_ix in range(self.num_agents):
                agent = self.agent_names[agent_ix]
                next_observations[ix, agent_ix] = obs_temp[agent]
                infos[ix, agent_ix] = infos_temp[agent]

        next_observations = next_observations[~dones, ...]
        subtracted_observations = next_observations - self.old_observations[~self.finished_envs, ...]
        self.old_observations[~self.finished_envs, ...] = next_observations
        #infos = infos[~dones, ...] <- this step goes in main code, because it is needed for
        if any(dones):
            assert len(subtracted_observations) < num_unfinished_envs, 'Next Observations has to decrease in size as environments are finished'
        return subtracted_observations, rewards, done_envs_cp, dones, infos

    def render(self):
        return np.concatenate([self.env_set[i].render(mode='rgb_array')[np.newaxis, ...] for i in range(0, self.size)], axis=0)

if __name__ == '__main__':
    env_params = {
        'n_pistons': 5, 'local_ratio': 1.0, 'time_penalty': 0.0, 'continuous': False,
        'random_drop': True, 'random_rotate': True, 'ball_mass': 0.75, 'ball_friction': 0.3,
        'ball_elasticity': 1.5, 'max_cycles': 200
    }

    env = Pong_Worker(10, env_params)
    print(env.reset().shape)