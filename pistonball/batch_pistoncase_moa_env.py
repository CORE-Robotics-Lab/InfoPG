from policy_pistoncase import MOAPolicyCASE
from policy_base import Experience
from typing import Optional, Dict, List
from alex_net import Encoder
from torch.distributions import Categorical
import torch
import torch.optim as optim
import numpy as np
from batch_envs import BatchEnv
import random
import os
import pickle
import datetime


class Piston5AgentCase(BatchEnv):
    def __init__(self, batch: int, env_params: Dict, seed=None):
        super(Piston5AgentCase, self).__init__('pistonball_v4', batch, env_params, seed)
        self.adj_matrix = self.adj_matrix.astype(np.bool)
        """self.adj_matrix = np.zeros((self.BATCH_SIZE, len(self.AGENT_NAMES), len(self.AGENT_NAMES)))
        for i in range(0, self.BATCH_SIZE):
            for j in range(0, len(self.AGENT_NAMES)):
                if j - 1 >= 0:
                    self.adj_matrix[i][j][j - 1] = 1
                else:
                    self.adj_matrix[i][j][j+2] = 1

                if j + 1 < len(self.AGENT_NAMES):
                    self.adj_matrix[i][j][j + 1] = 1
                else:
                    self.adj_matrix[i][j][j-2] = 1"""

    def update_adj_matrix(self):
        pass

    def loop(self, user_params: Dict, policies: Dict[str, MOAPolicyCASE], optimizers: Dict[str, torch.optim.Optimizer], schedulers):
        device = user_params['device']
        alex_encoder = Encoder(device)
        epochs = user_params['epochs']
        verbose = user_params['verbose']
        communicate = user_params['communicate']
        max_grad_norm = user_params['max_grad_norm']
        time_penalty = user_params['time_penalty']
        early_reward_benefit = user_params['early_reward_benefit']
        #k_levels = user_params['k-levels']

        summary_stats = []
        for epoch in range(0, epochs):
            if verbose:
                print('Epoch: %s' % (epoch))
            observations = self.batch_reset()
            prev_action_dists = torch.rand((self.BATCH_SIZE, self.N_AGENTS, 3))
            for step in range(self.MAX_CYCLES):
                num_left_batches = np.count_nonzero(self.DONE_ENVS == False)  # number of environments that aren't done
                left_batches = np.where(self.DONE_ENVS == False)[0]

                if num_left_batches == 0:
                    break

                memory = {}  # this is a map of agent_name -> a list of length self.BATCH_SIZE, but batches that are done are set to None
                for agent in self.AGENT_NAMES:
                    memory[agent] = []
                    for actual_batch_ix in range(0, self.BATCH_SIZE):
                        if not self.DONE_ENVS[actual_batch_ix]:
                            memory[agent].append(Experience())
                        else:
                            memory[agent].append(None)
                    assert len(memory[agent]) == self.BATCH_SIZE, 'Error Here'

                with torch.no_grad():
                    for agent in self.AGENT_NAMES:
                        observations[agent] = alex_encoder(observations[agent])

                info = {}
                for agent_ix, agent in enumerate(self.AGENT_NAMES):
                    input_messages = []
                    for batch_ix in range(0, num_left_batches):
                        actual_batch_ix = left_batches[batch_ix]
                        neighbor_mask = torch.tensor(self.adj_matrix[actual_batch_ix][agent_ix], dtype=torch.bool)
                        input_messages.append(prev_action_dists[actual_batch_ix, neighbor_mask].unsqueeze(0)) # -> appending 1, seq_len (neighbors), 3
                    input_prev_actions = torch.cat(input_messages, dim=0).to(device)
                    #input_prev_actions = input_prev_actions.permute(1,0,2) # batch, seq_len, size -> seq_len, batch, size
                    if agent == 'piston_2':
                        with torch.no_grad():
                            output_dist, output_value, output_actions_neighbors = policies[agent].forward(observations[agent], None, input_prev_actions)
                    else:
                        output_dist, output_value, output_actions_neighbors = policies[agent].forward(observations[agent], None, input_prev_actions)
                    info[agent] = [output_dist, output_value, output_actions_neighbors]

                actions = {agent_name: [-1 for _ in range(0, num_left_batches)] for agent_name in self.AGENT_NAMES}
                for agent_ix, agent in enumerate(self.AGENT_NAMES):
                    final_policy_distribution = info[agent][0].to('cpu')
                    distribution = Categorical(probs=final_policy_distribution)
                    batch_action = distribution.sample()
                    batched_log_prob = distribution.log_prob(batch_action)

                    for batch_ix in range(0, num_left_batches):
                        if agent == 'piston_2':
                            action = random.randint(0, 2) # piston 2 uses a randomized policy, no gradient
                        else:
                            action = batch_action[batch_ix].item()
                        log_prob = batched_log_prob[batch_ix]
                        actions[agent][batch_ix] = action

                        actual_batch_number = left_batches[batch_ix]
                        memory[agent][actual_batch_number].state_val = info[agent][1][batch_ix]
                        memory[agent][actual_batch_number].action = action
                        memory[agent][actual_batch_number].log_prob = log_prob
                        memory[agent][actual_batch_number].policy_distribution = final_policy_distribution[batch_ix]
                        prev_action_dists[actual_batch_number][agent_ix] = final_policy_distribution[batch_ix].detach()

                loss = torch.nn.CrossEntropyLoss(reduction='sum')
                for agent_ix, agent in enumerate(self.AGENT_NAMES):
                    actual_output_dist = torch.zeros(size=info[agent][2].shape[:-1], dtype=torch.long, device=device)
                    for batch_ix in range(0, num_left_batches):
                        actual_batch_ix = left_batches[batch_ix]
                        neighbor_names = self.AGENT_NAMES[self.adj_matrix[actual_batch_ix][agent_ix]]
                        for neighbor_ix, neighbor in enumerate(neighbor_names):
                            actual_output_dist[batch_ix, neighbor_ix] = actions[neighbor][batch_ix] # one hot encoding for cross entropy loss

                    for batch_ix in range(0, num_left_batches):
                        actual_batch_number = left_batches[batch_ix]
                        memory[agent][actual_batch_number].regularizer = loss(info[agent][2][batch_ix], actual_output_dist[batch_ix])

                next_observations, rewards, dones = self.batch_step(actions, step, time_penalty, early_reward_benefit)

                for agent in self.AGENT_NAMES:
                    for batch_ix in range(0, num_left_batches):
                        actual_batch_number = left_batches[batch_ix]
                        memory[agent][actual_batch_number].rewards = rewards[agent][batch_ix]
                    policies[agent].add_to_memory(memory[agent])
                self.update_adj_matrix()

                for agent in self.AGENT_NAMES:
                    observations[agent] = next_observations[agent][~dones[agent]]

            for agent in self.AGENT_NAMES:
                optimizers[agent].zero_grad(set_to_none=False)
                policies[agent].set_batched_storage(self.BATCH_SIZE)

            epoch_data = {}
            iterations = 0
            for agent in self.AGENT_NAMES:
                loss, batched_mean_reward, batched_length_iteration = policies[agent].compute_loss()
                mean_reward = batched_mean_reward.mean()
                mean_iteration_length = batched_length_iteration.mean()
                iterations += mean_iteration_length
                epoch_data[agent] = [mean_reward, mean_iteration_length, batched_mean_reward, batched_length_iteration]
                #print('Performing backprop on %s' % (agent))
                if verbose:
                    print('\t Reward for %s: %f' % (agent, mean_reward))
                if agent != 'piston_2':
                    loss.backward(retain_graph=True)
            summary_stats.append(epoch_data)
            if verbose:
                print('\t *Team Mean Iterations: %s' % (iterations/(self.N_AGENTS-1)))

            for agent in self.AGENT_NAMES:
                if agent == 'piston_2':
                    continue # no gradient for piston 2
                optimizers[agent].step()

            for agent in self.AGENT_NAMES:
                policies[agent].clear_memory()   
        return policies, optimizers, summary_stats

if __name__ == '__main__':
    if torch.cuda.is_available():
        device=torch.device("cuda:0")
        print('**Using nvidia tesla')
    else:
        device=torch.device("cpu")
        print('**Using: cpu')
    n_agents = 5
    max_cycles = 200
    encoding_size = 300
    policy_latent_size = 20
    action_space = 3
    lr = 0.001
    epochs = 1000
    batch = 2

    env_params = {
        'n_pistons': n_agents, 'local_ratio': 1.0, 'time_penalty': 0.0, 'continuous': False,
        'random_drop': True, 'random_rotate': True, 'ball_mass': 0.75, 'ball_friction': 0.3,
        'ball_elasticity': 1.5, 'max_cycles': max_cycles
    }

    user_params = {
        'device': device,
        'epochs': epochs,
        'verbose': True,
        'communicate': True,
        'max_grad_norm': 0.5,
        'time_penalty': 7e-3,
        'early_reward_benefit': 0.25,
        'consensus_update': False,
    }

    env = Piston5AgentCase(batch, env_params)
    policies = {agent: MOAPolicyCASE(device) for agent in env.get_agent_names()}
    optimizers = {agent: optim.Adam(policies[agent].parameters(), lr) for agent in env.get_agent_names()}
    
    policies, optimizers, summary_stats = env.loop(user_params, policies, optimizers, None)
    
    experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    path = os.path.join('experiments', 'case_test', experiment_name + 'moa')
    os.makedirs(path)

    with open('%s.pkl' % (os.path.join(path, 'data')), 'wb') as f:
        pickle.dump(summary_stats, f)
        
    for agent in policies.keys():
        torch.save({
            'policy': policies[agent].state_dict(),
            'optimizer': optimizers[agent].state_dict()
        }, os.path.join(path, agent+'.pt'))
