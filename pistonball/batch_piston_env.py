from batch_envs import BatchEnv
from typing import Dict
import torch
from policy_piston import PistonPolicy, MOAPolicy
from alex_net import Encoder
import numpy as np
from torch.distributions import Categorical
import torch.optim as optim

class PistonEnv(BatchEnv):
    def __init__(self, batch: int, env_params: Dict, seed=None):
        print('**Using Normal InfoPG Piston Env (for consensus, a2c, infopg)')
        super(PistonEnv, self).__init__('pistonball_v4', batch, env_params, seed)

    def update_adj_matrix(self):
        pass

    def loop(self, user_params: Dict, policies: Dict[str, PistonPolicy], optimizers: Dict[str, torch.optim.Optimizer], schedulers):
        """
        Executes training loop on batch environment
        :param user_params: dict of user params
        :param policies: dict of policies, mapping agent_name(str) -> Policy()
        :param optimizers: dict of optimizers, mapping agent_name(str) -> Optimizer()
        :return: policies, optimizers, summary_stats
        """

        device = user_params['device']
        alex_encoder = Encoder(device)
        epochs = user_params['epochs']
        verbose = user_params['verbose']
        communicate = user_params['communicate']
        max_grad_norm = user_params['max_grad_norm']
        time_penalty = user_params['time_penalty']
        early_reward_benefit = user_params['early_reward_benefit']
        if 'k-levels' in user_params.keys():
            k_levels = user_params['k-levels']
        else:
            k_levels = 1
        print('**Using K-Levels: ', k_levels)

        if 'consensus_update' in user_params.keys():
            consensus_update = user_params['consensus_update']
        else:
            consensus_update = False
        print('**Using Consensus-Update: ', consensus_update)

        summary_stats = []
        for epoch in range(0, epochs):
            if verbose:
                print('Epoch: %s' % (epoch))
            observations = self.batch_reset()
            for step in range(self.MAX_CYCLES):
                num_left_batches = np.count_nonzero(self.DONE_ENVS == False)  # number of environments that aren't done
                left_batches = np.where(self.DONE_ENVS == False)[0]  # indeces of batches that aren't done (length is num_left_batches)
                if num_left_batches == 0:
                    break

                memory = self.initialize_memory()

                with torch.no_grad():
                    for agent in self.AGENT_NAMES:
                        observations[agent] = alex_encoder(observations[agent])

                policy_initial = {}
                for agent in self.AGENT_NAMES:
                    initial_policy_distribution, state_val = policies[agent].forward(observations[agent], 0, None)
                    for batch_ix in range(0, num_left_batches):
                        memory[agent][left_batches[batch_ix]].state_val = state_val[batch_ix]
                    policy_initial[agent] = initial_policy_distribution

                actions = {agent_name: [-1 for _ in range(0, num_left_batches)] for agent_name in self.AGENT_NAMES}
                if communicate:
                    policy_initial = self.k_level_communication(policies, policy_initial, num_left_batches, left_batches, k_levels)

                for agent in self.AGENT_NAMES:
                    final_policy_distribution = policies[agent].forward(policy_initial[agent], 2, None).to('cpu')
                    distribution = Categorical(probs=final_policy_distribution)
                    batch_action = distribution.sample()
                    batched_log_prob = distribution.log_prob(batch_action)
                    for batch_ix in range(0, num_left_batches):
                        action = batch_action[batch_ix].item()
                        log_prob = batched_log_prob[batch_ix]
                        actions[agent][batch_ix] = action
                        actual_batch_number = left_batches[batch_ix]
                        memory[agent][actual_batch_number].action = action
                        memory[agent][actual_batch_number].log_prob = log_prob
                        memory[agent][actual_batch_number].policy_distribution = final_policy_distribution[batch_ix]

                next_observations, rewards, dones = self.batch_step(actions, step, time_penalty, early_reward_benefit)
                self.add_rewards_to_memory(policies, memory, rewards, num_left_batches, left_batches)
                observations = self.conclude_step(next_observations, dones)

            for agent in self.AGENT_NAMES:
                optimizers[agent].zero_grad(set_to_none=False)
                policies[agent].set_batched_storage(self.BATCH_SIZE)

            epoch_data, team_iterations = self.compute_epoch_data(policies, verbose=True)
            summary_stats.append(epoch_data)
            if verbose:
                print('\t *Team Mean Iterations: %s' % (team_iterations))

            for agent in self.AGENT_NAMES:
                torch.nn.utils.clip_grad_norm_(policies[agent].parameters(), max_grad_norm)

            self.conclude_epoch(policies, optimizers, schedulers)
            if consensus_update:
                self.consensus_update(policies)

        return policies, optimizers, summary_stats


class PistonEnv_MOA(BatchEnv):
    def __init__(self, batch: int, env_params: Dict, seed=None):
        super(PistonEnv_MOA, self).__init__('pistonball_v4', batch, env_params, seed)
        self.adj_matrix = self.adj_matrix.astype(np.bool)
        print('**Using MOA Piston Env (only for moa!)')
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

    def loop(self, user_params: Dict, policies: Dict[str, MOAPolicy], optimizers: Dict[str, torch.optim.Optimizer], schedulers):
        device = user_params['device']
        alex_encoder = Encoder(device)
        epochs = user_params['epochs']
        verbose = user_params['verbose']
        max_grad_norm = user_params['max_grad_norm']
        time_penalty = user_params['time_penalty']
        early_reward_benefit = user_params['early_reward_benefit']

        summary_stats = []
        for epoch in range(0, epochs):
            if verbose:
                print('Epoch: %s' % (epoch))
            observations = self.batch_reset()
            prev_action_dists = torch.rand((self.BATCH_SIZE, self.N_AGENTS, self.ACTION_SPACE))
            for step in range(self.MAX_CYCLES):
                num_left_batches = np.count_nonzero(self.DONE_ENVS == False)  # number of environments that aren't done
                left_batches = np.where(self.DONE_ENVS == False)[0]
                if num_left_batches == 0:
                    break

                memory = self.initialize_memory()

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
                    output_dist, output_value, output_actions_neighbors = policies[agent].forward(observations[agent], None, input_prev_actions)
                    info[agent] = [output_dist, output_value, output_actions_neighbors]

                actions = {agent_name: [-1 for _ in range(0, num_left_batches)] for agent_name in self.AGENT_NAMES}
                for agent_ix, agent in enumerate(self.AGENT_NAMES):
                    final_policy_distribution = info[agent][0].to('cpu')
                    distribution = Categorical(probs=final_policy_distribution)
                    batch_action = distribution.sample()
                    batched_log_prob = distribution.log_prob(batch_action)

                    for batch_ix in range(0, num_left_batches):
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
                            actual_output_dist[batch_ix, neighbor_ix] = actions[neighbor][batch_ix] #action taken by some agent==neighbor_ix, and at some batch_ix

                    for batch_ix in range(0, num_left_batches):
                        actual_batch_number = left_batches[batch_ix]
                        memory[agent][actual_batch_number].regularizer = loss(info[agent][2][batch_ix], actual_output_dist[batch_ix])

                next_observations, rewards, dones = self.batch_step(actions, step, time_penalty, early_reward_benefit)
                self.add_rewards_to_memory(policies, memory, rewards, num_left_batches, left_batches)
                observations = self.conclude_step(next_observations, dones)

            for agent in self.AGENT_NAMES:
                optimizers[agent].zero_grad(set_to_none=False)
                policies[agent].set_batched_storage(self.BATCH_SIZE)

            epoch_data, team_iterations = self.compute_epoch_data(policies, verbose=True)
            summary_stats.append(epoch_data)
            if verbose:
                print('\t *Team Mean Iterations: %s' % (team_iterations))

            for agent in self.AGENT_NAMES:
                torch.nn.utils.clip_grad_norm_(policies[agent].parameters(), max_grad_norm)

            self.conclude_epoch(policies, optimizers, schedulers)

        return policies, optimizers, summary_stats

if __name__ == '__main__':
    """n_agents = 5
    max_cycles = 10
    encoding_size = 300
    policy_latent_size = 20
    action_space = 3
    lr = 0.001
    device = 'cpu'
    batch = 5
    env_params = {
        'n_pistons': n_agents, 'local_ratio': 1.0, 'time_penalty': 0.0, 'continuous': False,
        'random_drop': True, 'random_rotate': True, 'ball_mass': 0.75, 'ball_friction': 0.3,
        'ball_elasticity': 1.5, 'max_cycles': max_cycles
    }
    user_params = {
        'device': device,
        'epochs': 1,
        'verbose': True,
        'communicate': True,
        'max_grad_norm': 0.5,
        'time_penalty': -0.01,
        'early_reward_benefit': 1.5,
        'consensus_update': False,
    }

    env = PistonEnv(batch, env_params)
    policies = {agent: PistonPolicy(encoding_size, policy_latent_size, action_space, device) for agent in env.get_agent_names()}
    optimizers = {agent: optim.Adam(policies[agent].parameters(), lr) for agent in env.get_agent_names()}

    env.loop(user_params, policies, optimizers, None)"""

    n_agents = 5
    max_cycles = 10
    encoding_size = 300
    policy_latent_size = 20
    action_space = 3
    lr = 0.001
    device = 'cpu'
    batch = 5
    env_params = {
        'n_pistons': n_agents, 'local_ratio': 1.0, 'time_penalty': 0.0, 'continuous': False,
        'random_drop': True, 'random_rotate': True, 'ball_mass': 0.75, 'ball_friction': 0.3,
        'ball_elasticity': 1.5, 'max_cycles': max_cycles
    }
    user_params = {
        'device': device,
        'epochs': 2,
        'verbose': True,
        'max_grad_norm': 0.5,
        'time_penalty': -0.01,
        'early_reward_benefit': 1.5,
    }

    env = PistonEnv_MOA(batch, env_params)
    policies = {agent: MOAPolicy(encoding_size, policy_latent_size, action_space, device) for agent in env.get_agent_names()}
    optimizers = {agent: optim.Adam(policies[agent].parameters(), lr) for agent in env.get_agent_names()}
    env.loop(user_params, policies, optimizers, None)