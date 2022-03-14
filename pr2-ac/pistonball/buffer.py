import numpy as np
from collections import deque

class Batch_Storage:
    def __init__(self, batch_size, capacity=100000):
        self.batch_size = batch_size
        self.batch_storage = [deque([], capacity) for _ in range(0, self.batch_size)]
        self.batch_indeces = np.arange(0, self.batch_size).astype(np.uint8)
        self.data_items = 5
        self.random_sample_size = 50

    def add_to_batch(self, dones, observations, executed_actions, rewards, next_observations, done_envs_curr):
        assert len(dones) == self.batch_size, 'Error here'
        dones = np.array(dones)
        num_left = len(np.argwhere(dones == False).flatten())
        assert np.array([len(observations) == num_left, len(rewards) == num_left, len(next_observations) == num_left,
                         len(executed_actions == num_left), len(done_envs_curr == num_left)]).all(), \
            '%s, %s, %s, %s, %s != %s' % (observations.shape, rewards.shape, next_observations.shape, executed_actions.shape, done_envs_curr.shape, num_left)
        add_indeces = self.batch_indeces[~dones]

        for temp_ix, add_index in enumerate(add_indeces):
            self.batch_storage[add_index].append([observations[temp_ix], executed_actions[temp_ix],
                                                  rewards[temp_ix], next_observations[temp_ix], done_envs_curr[temp_ix]])

    def print_data(self):
        for env_ix in range(self.batch_size):
            print('Number of timesteps: ', len(self.batch_storage[env_ix]))
            for t in range(0, len(self.batch_storage[env_ix])):
                print('At time: ', t)
                print([self.batch_storage[env_ix][t][data_ix].shape for data_ix in range(0, self.data_items)])

    def get_data(self):
        def compact_data(env_ix, data_ix):
            return np.concatenate([self.batch_storage[env_ix][t][data_ix][np.newaxis, ...] for t in
                                   np.random.choice(np.arange(0, len(self.batch_storage[env_ix])), size=self.random_sample_size)], axis=0)
        return [[compact_data(env_ix, data_ix) for data_ix in range(0, self.data_items)] for env_ix in range(0, self.batch_size)]

    def ready_for_sending(self):
        return all([len(batch) >= self.random_sample_size for batch in self.batch_storage])

    def clear_data(self):
        self.batch_storage = [[] for _ in range(0, self.batch_size)]