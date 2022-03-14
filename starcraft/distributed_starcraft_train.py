import math

from distributed_starcraft_policy import Combined_StarCraft_Helper
import ray
from distributed_starcraft_env import StarCraft_Worker, get_nagents_action_space
import numpy as np
from util import validate_state_dicts
import itertools
import torch.optim as optim
import torch.nn as nn
import torch
import os
import datetime
import sys

class Runner:
    def __init__(self, map_name, rewards_positive, batch_size, num_workers = 4):
        self.env_params = {
            'map_name': map_name, 'reward_only_positive': rewards_positive
        }
        assert batch_size % num_workers == 0
        self.BATCH_SIZE = batch_size
        self.BATCH_SPLIT_SIZE = self.BATCH_SIZE//num_workers
        self.NUM_WORKERS = num_workers
        self.N_AGENTS, self.ACTION_SPACE = get_nagents_action_space(self.env_params)
        self.max_grad_norm = 6.0
        ray.init(num_cpus = self.NUM_WORKERS, _node_ip_address="0.0.0.0")
        self.envs = [StarCraft_Worker.remote(self.BATCH_SPLIT_SIZE, self.env_params, time_penalty=0.0, device='cpu') for _ in range(self.NUM_WORKERS)]

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
        self.envs = [StarCraft_Worker.remote(self.BATCH_SPLIT_SIZE, self.env_params, time_penalty=0.0, device='cpu') for _ in range(self.NUM_WORKERS)]

    def set_curriculum(self):
        ray.get([worker.curriculum_step.remote() for worker in self.envs])

    def train_loop(self, save_path, k=1, adv='normal', use_critic=False, consensus=False, transfer_experiment=None):
        if torch.cuda.is_available():
            device=torch.device("cuda")
            num_gpus = 1
            print('**Using:  for inference', torch.cuda.get_device_name(device))
        else:
            device=torch.device("cpu")
            num_gpus=0
            print('**Using: cpu for inference')

        model = Combined_StarCraft_Helper(device, self.env_params['map_name'], k_levels=k)
        optimizers = [optim.Adam(param, lr=1e-4) for param in model.parameters()]
        if transfer_experiment is not None:
            print('Transferring from: ', transfer_experiment)
            with open(os.path.join('experiments', 'starcraft', transfer_experiment, 'combined_model.pt'), 'rb') as f:
                d = torch.load(f, map_location=device)
            model.load_state_dicts(d['policy'])
            for ix in range(0, self.N_AGENTS):
                optimizers[ix].load_state_dict(d['optimizer'][ix])

        epochs = 2000
        analytics = None
        best_performance = float('-inf')
        for epoch in range(0, epochs):
            print('Epoch %s/%s' % (epoch, epochs))
            self.send_weights(model)
            if epoch % 10 == 0 and epoch != 0:
                self.set_curriculum()
            if not self.assert_all_same():
                raise Exception('Error with setting model weights across workers')
            try:
                data = ray.get([worker.train.remote() for worker in self.envs])
            except Exception as e:
                self.reset()
                for agent_ix in range(0, self.N_AGENTS):
                    optimizers[agent_ix].zero_grad()
                print('*GOING TO RESET!*')
                print('*Because of: %s' % (str(e)))
                #self.reset()
                print('*DONE WITH RESET!*')
                epoch -= 1
                continue

            data = list(itertools.chain.from_iterable(data))

            for agent_ix in range(0, self.N_AGENTS):
                optimizers[agent_ix].zero_grad()

            agent_statistics = self.compute_gradients(self.get_agent_data(model, data, device), device,
                                                      adv_opt=adv, use_critic=use_critic, gamma=0.99)

            epoch_returns = np.zeros((1, self.N_AGENTS, 2, self.BATCH_SIZE))
            mutual_info = np.zeros((1, self.N_AGENTS, self.BATCH_SIZE, 2))
            team_total = 0
            for agent_ix in range(self.N_AGENTS):
                agent_loss, mean_rewards_over_batch, length_iteration_over_batch = agent_statistics[agent_ix]
                epoch_returns[0, agent_ix, 0] = mean_rewards_over_batch
                epoch_returns[0, agent_ix, 1] = length_iteration_over_batch
                agent_loss.backward(retain_graph=True)
                team_total += mean_rewards_over_batch.mean().item()
                print('**Agent %s Return over Batches: %s' % (agent_ix, mean_rewards_over_batch.mean()))
                print('**Agent %s Survival Length over Batches: %s' % (agent_ix, length_iteration_over_batch.mean()))

            for param_set in model.parameters():
                torch.nn.utils.clip_grad_norm_(param_set, self.max_grad_norm)

            for agent_ix in range(0, self.N_AGENTS):
                optimizers[agent_ix].step()

            if analytics is None:
                analytics = epoch_returns
            else:
                analytics = np.append(analytics, epoch_returns, axis=0)

            if consensus:
                model.consensus_update()

            if save_path is not None:
                self.save_checkpoint(model, optimizers, analytics, save_path)
                if team_total > best_performance:
                    best_performance = team_total
                    self.save_checkpoint(model, optimizers, analytics, os.path.join(save_path, 'best'))

    def save_checkpoint(self, model, optimizer, analytics, path):
        torch.save({
            'policy': model.state_dicts(),
            'optimizer': [op.state_dict() for op in optimizer]
        }, os.path.join(path, 'combined_model.pt'))

        np.save(os.path.join(path, 'data.npy'), analytics)

    def get_agent_data(self, model, data, device):
        data_per_agent = [[] for _ in range(0, self.N_AGENTS)]
        for batch_ix, batch in enumerate(data):
            input_obs, actions, rewards, available_actions, adj_matrix_cop = batch
            num_iterations = len(input_obs)

            adj_matrix = np.tile((~np.eye((self.N_AGENTS))[np.newaxis, ...].astype(bool)), (num_iterations, 1, 1))
            observation_list = [torch.tensor(input_obs[:, agent_ix], dtype=torch.float32).to(device) for agent_ix in range(0, self.N_AGENTS)]
            available_actions_list = [torch.tensor(available_actions[:, agent_ix], dtype=torch.bool).to(device) for agent_ix in range(0, self.N_AGENTS)]
            adj_matrix_torch = torch.tensor(adj_matrix, dtype=torch.bool).to(device)
            output_action_dist, state_vals, _ = model(observation_list, adj_matrix_torch, available_actions_list)
            reformatted_action_dist = torch.cat([dist.unsqueeze(1) for dist in output_action_dist], dim=1) # batch x agents x 3
            torch_dist = torch.distributions.Categorical(probs=reformatted_action_dist)
            log_probs = torch_dist.log_prob(torch.tensor(actions, dtype=torch.int).to(device))
            entropy = torch_dist.entropy()

            for agent_ix in range(0, self.N_AGENTS):
                agent_mask = torch.tensor(available_actions[:, agent_ix, 0] == False, dtype=torch.bool)
                #print('on %s, agent %s, survived: %s/%s' % (batch_ix, agent_ix, torch.count_nonzero(agent_mask), len(agent_mask)))
                data_per_agent[agent_ix].append([log_probs[agent_mask, agent_ix].unsqueeze(1), state_vals[agent_ix][agent_mask],
                                                 rewards[agent_mask, agent_ix], entropy[agent_mask, agent_ix].unsqueeze(1)])
        return data_per_agent

    def compute_gradients(self, data_per_agent, device, adv_opt='normal', use_critic=False, gamma=0.99):
        agent_information = []
        for agent_ix in range(0, self.N_AGENTS):
            mean_reward = np.zeros((self.BATCH_SIZE))
            length_iteration = np.zeros((self.BATCH_SIZE))
            loss = []
            assert len(data_per_agent[agent_ix]) == self.BATCH_SIZE, 'Error with previous loop'
            for batch_ix in range(0, self.BATCH_SIZE):
                experience_list = data_per_agent[agent_ix][batch_ix]
                log_probs, state_vals, rewards, _ = experience_list
                length_of_iteration_for_batch = len(rewards)
                R = 0
                returns = []
                l1_loss_func = nn.SmoothL1Loss()
                for ix in range(length_of_iteration_for_batch -1, -1, -1):
                    R = rewards[ix] + R*gamma
                    returns.append(R)
                returns.reverse()
                returns = torch.tensor(returns, dtype=torch.float32).to(device)
                mean_rewards_for_batch = returns.mean().item()
                returns = (returns - returns.mean())/(returns.std() + 1e-5)
                if adv_opt == 'normal':
                    adv = returns - state_vals
                    actor_loss = torch.mean(-log_probs*(adv), dim = 0)
                    critic_loss = torch.mean(l1_loss_func(state_vals, returns), dim = 0)
                    loss_for_batch = actor_loss + critic_loss
                elif adv_opt == 'info':
                    adv = returns.clamp(min=0)
                    actor_loss = torch.mean(-log_probs*(adv), dim = 0)
                    loss_for_batch = actor_loss
                    if use_critic:
                        loss_for_batch += torch.mean(l1_loss_func(state_vals, returns), dim = 0)
                else:
                    raise Exception(adv_opt + ' isnt a valid option, should be normal or info!')
                mean_reward[batch_ix] = mean_rewards_for_batch
                length_iteration[batch_ix] = length_of_iteration_for_batch
                loss.append(loss_for_batch)
            agent_loss = torch.cat(loss).mean()
            agent_information.append([agent_loss, mean_reward, length_iteration])
        return agent_information

    def close(self):
        ray.get([worker.close.remote() for worker in self.envs])
        ray.shutdown()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', type=int, required=True, help="Batch Size")
    parser.add_argument('-workers', type=int, required=True, help="Number of CPUS")
    parser.add_argument('-map', type=str, required=True, default='3m', help="3m or 2c_vs_64zg")
    parser.add_argument('-k', type=int, required=True, default=1, help="Number of K Levels")
    parser.add_argument('-adv', type=str, required=True, default='normal', help="Use Normal Advantage or Info Adv, options = 'normal' or 'info'")

    parser.add_argument('-positive_rewards', action='store_true', required=False, default=False, help="Use only positive reward or negatives too?" )
    parser.add_argument('-transfer_experiment', type=str, required=False, default=None, help='Continue learning from some previous experiment?')
    parser.add_argument('-critic', action='store_true', required=False, default=False, help="Use Critic in StarCraft?")
    parser.add_argument('-consensus', action='store_true', required=False, default=False, help="Using Consensus Update")
    parser.add_argument('-eval', action='store_true', required=False, default=False, help="Eval Mode?")
    args = parser.parse_args()

    workers = args.workers

    run = Runner(args.map, args.positive_rewards, args.batch, workers)
    if not args.eval:
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        path = os.path.join('experiments', 'pr2', experiment_name)
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'best'))

        with open(os.path.join(path, 'params.txt'), 'w') as f:
            f.write('Args: batch %s, workers %s, adv %s, critic %s' % (
                args.batch, args.workers, args.adv, args.critic,
            ))
        try:
            run.train_loop(path, k= args.k, adv=args.adv, use_critic=args.critic, consensus=args.consensus, transfer_experiment=args.transfer_experiment)
        except Exception as e:
            run.close()
            raise e
