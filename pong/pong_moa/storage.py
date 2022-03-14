import numpy as np
import torch.nn as nn
import torch

class Batch_Storage:
    def __init__(self, batch_size, workers):
        self.batch_size = batch_size
        self.workers = workers
        self.batch_split_size = int(batch_size/workers)
        self.batch_storage = [[] for _ in range(0, self.batch_size)]
        self.batch_split_indeces = np.arange(0, self.batch_split_size).astype(np.uint8)

    def add_to_batch(self, worker_ix, dones, log_probs, state_vals, rewards, moa_loss):
        assert len(dones) == self.batch_split_size, 'Error here'
        start_ix = worker_ix*self.batch_split_size
        dones = np.array(dones)
        num_left = len(np.argwhere(dones == False).flatten())
        assert np.array([len(log_probs) == num_left, len(state_vals) == num_left, len(rewards) == num_left]).all(),\
            '%s, %s, %s != %s' % (log_probs.shape, state_vals.shape, rewards.shape, num_left)
        add_indeces = self.batch_split_indeces[~dones]

        for temp_ix, add_index in enumerate(add_indeces):
            self.batch_storage[start_ix+add_index].append([log_probs[temp_ix], state_vals[temp_ix], rewards[temp_ix], moa_loss[temp_ix]])

    def compute_loss(self, device, gamma=0.99):
        mean_reward = np.zeros((self.batch_size))
        length_iteration = np.zeros((self.batch_size))
        loss = []
        for batch_ix in range(0, self.batch_size):
            experience_list = self.batch_storage[batch_ix]
            length_of_iteration_for_batch = len(experience_list)
            R = 0
            returns = []
            l1_loss_func = nn.SmoothL1Loss()
            for r in experience_list[::-1]:
                R = r[2] + gamma * R
                returns.append(R)

            returns.reverse()
            returns = torch.tensor(returns, dtype=torch.float32).to(device)
            mean_rewards_for_batch = returns.mean().item()
            returns = (returns - returns.mean())/(returns.std() + 1e-5)
            log_probs = torch.vstack([experience[0] for experience in experience_list]).to(device)
            state_vals = torch.vstack([experience[1] for experience in experience_list]).to(device)
            moa_loss = torch.vstack([experience[3] for experience in experience_list]).to(device)
            adv = (returns + 0.1*moa_loss) - state_vals
            actor_loss = torch.mean(-log_probs*(adv), dim = 0)
            critic_loss = torch.mean(l1_loss_func(state_vals, returns), dim = 0)
            loss_for_batch = actor_loss + critic_loss
            mean_reward[batch_ix] = mean_rewards_for_batch
            length_iteration[batch_ix] = length_of_iteration_for_batch
            loss.append(loss_for_batch)
        agent_loss = torch.cat(loss).mean()
        return agent_loss, mean_reward, length_iteration

if __name__ == '__main__':
    storage = Batch_Storage(16, 4)
    d = [1,2]
    storage.add_to_batch(2, dones=np.array([False, True, False, True], dtype=bool), log_probs=d, state_vals=d, rewards=d)
    print(storage.batch_storage)
    #storage.add_to_batch(2)
    #print(storage.batch_split_size, storage.batch_split_indeces)
