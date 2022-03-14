import math

from pr2_policy import Combined_Piston_Helper
import ray
from distributed_pistonball_env import Piston_Worker, get_nagents_action_space
import numpy as np
import itertools
import torch.optim as optim
import torch.nn as nn
import torch
import os
import datetime
from util import validate_state_dicts
from alex_net_pistonball import Encoder

class Runner:
    def __init__(self, batch_size, num_workers = 4):
        self.env_params = {
            'n_pistons': 5, 'local_ratio': 1.0, 'time_penalty': 0.0, 'continuous': False,
            'random_drop': True, 'random_rotate': True, 'ball_mass': 0.75, 'ball_friction': 0.3,
            'ball_elasticity': 1.5, 'max_cycles': 200
        }
        assert batch_size % num_workers == 0
        self.BATCH_SIZE = batch_size
        self.BATCH_SPLIT_SIZE = self.BATCH_SIZE//num_workers
        self.NUM_WORKERS = num_workers
        self.N_AGENTS, self.ACTION_SPACE = get_nagents_action_space()
        self.max_grad_norm = 8.0
        ray.init(num_cpus = self.NUM_WORKERS, _node_ip_address="0.0.0.0")
        self.envs = [Piston_Worker.remote(self.BATCH_SPLIT_SIZE, self.env_params, time_penalty=7e-3, device='cpu') for _ in range(self.NUM_WORKERS)]

    def send_weights(self, model):
        state_dicts = model.state_dicts()
        new_state_dicts = [{} for _ in range(len(state_dicts))]
        for agent_ix in range(0, len(state_dicts)):
            for param_name in state_dicts[agent_ix].keys():
                if isinstance(state_dicts[agent_ix][param_name], torch.Tensor):
                    new_state_dicts[agent_ix][param_name] = state_dicts[agent_ix][param_name].clone().to('cpu')
        weight_id = ray.put(new_state_dicts)
        ray.get([worker.set_model_weights.remote(weight_id) for worker in self.envs])

    def assert_all_same(self):
        worker_state_dicts = ray.get([env.get_model_weights.remote() for env in self.envs])
        starting = worker_state_dicts[0]
        for i in range(1, self.NUM_WORKERS):
            if len(starting) != len(worker_state_dicts[i]):
                return False
            else:
                for j in range(0, len(starting)):
                    if not validate_state_dicts(starting[j], worker_state_dicts[i][j]):
                        return False
        return True

    def reset(self):
        self.close()
        ray.init(num_cpus = self.NUM_WORKERS, num_gpus=1, _node_ip_address="0.0.0.0")
        self.envs = [Piston_Worker.remote(self.BATCH_SPLIT_SIZE, self.env_params, time_penalty=0.0, device='cpu') for _ in range(self.NUM_WORKERS)]

    def set_curriculum(self):
        ray.get([worker.curriculum_step.remote() for worker in self.envs])

    def get_trajectory_eval(self, gamma=0.99):
        temp_data = ray.get([worker.eval.remote() for worker in self.envs])

        rewards_data = list(itertools.chain.from_iterable(temp_data))
        assert len(rewards_data) == self.BATCH_SIZE, 'something is wrong here'
        agent_information = [[] for _ in range(0, self.N_AGENTS)]
        for batch_ix in range(0, self.BATCH_SIZE):
            R = 0
            returns = []
            for ix in range(len(rewards_data[batch_ix]) -1, -1, -1):
                R = rewards_data[batch_ix][ix] + R*gamma
                returns.append(R)
            returns.reverse()
            returns = torch.tensor(returns, dtype=torch.float32).to('cpu')
            mean_rewards_for_batch = returns.mean(dim=0)
            for agent_ix in range(0, self.N_AGENTS):
                agent_information[agent_ix].append(mean_rewards_for_batch[agent_ix].item())
        return np.expand_dims(np.array(agent_information), axis=1)[np.newaxis, ...]


    def train_loop(self, save_path, transfer_experiment=None):
        if torch.cuda.is_available():
            device = torch.device('cuda:2')
            print(f"**Using: {torch.cuda.get_device_name(device)}, properties: {2}  for inference")
            torch.cuda.empty_cache()
        else:
            device=torch.device("cpu")
            num_gpus=0
            print('**Using: cpu for inference')

        model = Combined_Piston_Helper(device)
        optimizers = [optim.Adam(param, lr=1e-3) for param in model.parameters()]
        if transfer_experiment is not None:
            print('Transferring from: ', transfer_experiment)
            with open(os.path.join('experiments', 'starcraft', transfer_experiment, 'combined_model.pt'), 'rb') as f:
                d = torch.load(f, map_location=device)
            model.load_state_dicts(d['policy'])
            for ix in range(0, self.N_AGENTS):
                optimizers[ix].load_state_dict(d['optimizer'][ix])

        epochs = 1000
        analytics = None
        trajectory_analytics = None

        best_performance = float('-inf')
        for epoch in range(0, epochs):
            print('Epoch %s/%s' % (epoch, epochs))
            self.send_weights(model)
            if not self.assert_all_same():
                raise Exception('Error with setting model weights across workers')
            data = ray.get([worker.train.remote() for worker in self.envs])
            trajectories = self.get_trajectory_eval()

            if trajectory_analytics is None:
                trajectory_analytics = trajectories
            else:
                trajectory_analytics = np.append(trajectory_analytics, trajectories, axis=0)

            data = list(itertools.chain.from_iterable(data))
            for agent_ix in range(0, self.N_AGENTS):
                optimizers[agent_ix].zero_grad()

            actor, critic, kernel = self.get_agent_data(model, data, device)
            epoch_returns = np.zeros((1, self.N_AGENTS, 3))
            team_total = 0
            for agent_ix in range(self.N_AGENTS):
                actor_agent, critic_agent, kernel_agent = actor[agent_ix], critic[agent_ix], kernel[agent_ix]
                epoch_returns[0, agent_ix, 0] = actor_agent.item()
                epoch_returns[0, agent_ix, 1] = critic_agent.item()
                epoch_returns[0, agent_ix, 2] = kernel_agent.item()

                print('*Agent %s Actor Loss: %s' % (agent_ix, actor_agent.item()))
                print('*Agent %s Critic Loss: %s' % (agent_ix, critic_agent.item()))
                print('*Agent %s Kernel Loss: %s' % (agent_ix, kernel_agent.item()))
                (actor_agent+critic_agent+kernel_agent).backward(retain_graph=True)
                team_total += epoch_returns[0, agent_ix].sum()

            for param_set in model.parameters():
                torch.nn.utils.clip_grad_norm_(param_set, self.max_grad_norm)

            for agent_ix in range(0, self.N_AGENTS):
                optimizers[agent_ix].step()

            if analytics is None:
                analytics = epoch_returns
            else:
                analytics = np.append(analytics, epoch_returns, axis=0)

            if save_path is not None:
                self.save_checkpoint(model, optimizers, analytics, trajectory_analytics, save_path)
                if team_total < best_performance:
                    best_performance = team_total
                    self.save_checkpoint(model, optimizers, analytics, trajectory_analytics, os.path.join(save_path, 'best'))

    def save_checkpoint(self, model, optimizer, analytics, trajectory_analytics, path):
        torch.save({
            'policy': model.state_dicts(),
            'optimizer': [op.state_dict() for op in optimizer]
        }, os.path.join(path, 'combined_model.pt'))
        np.save(os.path.join(path, 'data.npy'), analytics)
        np.save(os.path.join(path, 'trajectories_data.npy'), trajectory_analytics)

    def get_agent_data(self, model: Combined_Piston_Helper, data, device):
        encoder = Encoder(device)
        gamma = 0.99
        num_particles = 16
        actor_loss_per_agent = [[] for _ in range(0, self.N_AGENTS)]
        critic_loss_per_agent = [[] for _ in range(0, self.N_AGENTS)]
        kernel_loss_per_agent = [[] for _ in range(0, self.N_AGENTS)]
        mse_loss = torch.nn.MSELoss()
        kernel_update_ratio = 0.5

        for batch_ix, batch in enumerate(data):
            input_obs, actions, rewards, next_obs, dones = batch
            #print(input_obs.shape, actions.shape, rewards.shape, next_obs.shape, dones.shape)

            encodings = [torch.tensor(input_obs[:, agent_ix]).to(device) for agent_ix in range(0, self.N_AGENTS)]
            curr_action_dists = model(encodings, None)
            reformatted_action_dist = torch.cat([dist.unsqueeze(1) for dist in curr_action_dists], dim=1) # batch x agents x 3
            torch_dist = torch.distributions.Categorical(probs=reformatted_action_dist)
            log_probs = torch_dist.log_prob(torch.tensor(actions, dtype=torch.int).to(device)).unsqueeze(-1)

            next_encodings = [torch.tensor(next_obs[:, agent_ix]).to(device) for agent_ix in range(0, self.N_AGENTS)]
            next_action_dists = model(next_encodings, None)

            for agent_ix in range(self.N_AGENTS):
                real_action_others = self.get_neighbors(curr_action_dists, agent_ix)
                rewards_agent = torch.tensor(rewards[:, agent_ix], dtype=torch.float32).to(device)
                predicted_actions_others = model.get_moa(next_encodings[agent_ix], curr_action_dists[agent_ix], num_particles, agent_ix)
                #print(real_action_others.shape, predicted_actions_others.shape)
                y_agent = torch.zeros_like(rewards_agent).to(device)
                q_total_others = torch.logsumexp(model.get_jointq(
                    next_encodings[agent_ix].unsqueeze(1).repeat(1, num_particles, 1),
                    next_action_dists[agent_ix].unsqueeze(1).repeat(1, num_particles, 1),
                    predicted_actions_others, agent_ix), dim=1)
                y_agent[~dones] = rewards_agent[~dones] + gamma*q_total_others[~dones]
                #print(encodings[agent_ix].shape, curr_action_dists[agent_ix].shape, torch.flatten(real_action_others, start_dim=-2, end_dim=-1).shape)
                actual_q_total = model.get_jointq(
                    encodings[agent_ix],
                    curr_action_dists[agent_ix],
                    torch.flatten(real_action_others, start_dim=-2, end_dim=-1), agent_ix)
                critic_loss = mse_loss(y_agent.detach(), actual_q_total).unsqueeze(0)
                q_targets = torch.logsumexp(model.get_jointq(encodings[agent_ix].unsqueeze(1).repeat(1, num_particles, 1),
                                                             curr_action_dists[agent_ix].unsqueeze(1).repeat(1, num_particles, 1),
                                                             predicted_actions_others, agent_ix), dim=1) - math.log(num_particles) + self.ACTION_SPACE * np.log(2)
                actor_loss = (-log_probs[:, agent_ix]*q_targets).mean().unsqueeze(0)

                n_update_actions = int(kernel_update_ratio*num_particles)
                n_fixed_actions = num_particles - n_update_actions

                fixed_actions = predicted_actions_others[:, :n_fixed_actions]
                update_actions = predicted_actions_others[:, n_fixed_actions+1:]

                svgd_target_values = model.get_jointq(
                    encodings[agent_ix].unsqueeze(1).repeat(1, n_fixed_actions, 1),
                    curr_action_dists[agent_ix].unsqueeze(1).repeat(1, n_fixed_actions, 1),
                    fixed_actions, agent_ix) # dim batchxn_fixed_actionsx1
                #squash_correction = torch.sum(torch.log(1 - fixed_actions**2), dim=-1)
                #svgd_target_values += squash_correction

                kernel_loss = 0
                for neighbor_ix in range(0, 1): # should be num_neighbors
                    ix = neighbor_ix*self.ACTION_SPACE
                    xs = fixed_actions[:, :, ix:ix+self.ACTION_SPACE].contiguous()
                    ys = update_actions[:, :, ix:ix+self.ACTION_SPACE].contiguous()
                    kernel_out = torch.cdist(xs, ys).sum(dim=-1).unsqueeze(-1) #dim = batch x n_fixed_action x n_update_actions
                    magnitude = kernel_out.clone().detach()
                    kernel_loss += svgd_target_values*magnitude + kernel_out
                actor_loss_per_agent[agent_ix].append(actor_loss)
                critic_loss_per_agent[agent_ix].append(critic_loss)
                kernel_loss_per_agent[agent_ix].append(kernel_loss)
        avg_actor_loss_per_agent = [torch.cat(actor_loss_per_agent[agent_ix]).mean() for agent_ix in range(self.N_AGENTS)]
        avg_critic_loss_per_agent = [torch.cat(critic_loss_per_agent[agent_ix]).mean() for agent_ix in range(self.N_AGENTS)]
        avg_kernel_loss_per_agent = [torch.cat(kernel_loss_per_agent[agent_ix]).mean() for agent_ix in range(self.N_AGENTS)]
        return avg_actor_loss_per_agent, avg_critic_loss_per_agent, avg_kernel_loss_per_agent

    def get_neighbors(self, curr_action_dists, agent_ix):
        neighbors = []
        if agent_ix == 0:
            neighbors.append(
                curr_action_dists[1].clone().detach().unsqueeze(1)
            )
        elif agent_ix == len(curr_action_dists) - 1:
            neighbors.append(
                curr_action_dists[-2].clone().detach().unsqueeze(1)
            )
        else:
            neighbors.append(
                curr_action_dists[agent_ix - 1].clone().detach().unsqueeze(1)
            )
            neighbors.append(
                curr_action_dists[agent_ix + 1].clone().detach().unsqueeze(1)
            )
        return torch.cat(neighbors, dim=1)

    def close(self):
        ray.get([worker.close.remote() for worker in self.envs])
        ray.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', type=int, required=True, help="Batch Size")
    parser.add_argument('-workers', type=int, required=True, help="Number of CPUS")

    parser.add_argument('-transfer_experiment', type=str, required=False, default=None, help='Continue learning from some previous experiment?')
    parser.add_argument('-eval', action='store_true', required=False, default=False, help="Eval Mode?")
    args = parser.parse_args()

    workers = args.workers

    run = Runner(args.batch, workers)
    if not args.eval:
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        path = os.path.join('experiments', 'pr2', experiment_name)
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'best'))

        with open(os.path.join(path, 'params.txt'), 'w') as f:
            f.write('Args: batch %s, workers %s' % (
                args.batch, args.workers
            ))
        try:
            run.train_loop(path, transfer_experiment=args.transfer_experiment)
        except Exception as e:
            run.close()
            raise e
