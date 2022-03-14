from distributed_pong_env import Pong_Worker
from distributed_pong_moa_policy import Combined_MOAPong_Helper
import torch.optim as optim
import torch
import numpy as np
import ray
import pickle
import os
from torch.distributions import Categorical
from alex_net_pong import Encoder
import datetime
from storage import Batch_Storage
import torch.nn as nn


class Runner:
    def __init__(self, batch_size, num_workers = 4):
        env_params = {
            'ball_speed': 15, 'left_paddle_speed': 13, 'right_paddle_speed': 13,
            'cake_paddle': False, 'max_cycles': 300, 'bounce_randomness': False
        }
        assert batch_size % num_workers == 0
        self.BATCH_SIZE = batch_size
        self.BATCH_SPLIT_SIZE = self.BATCH_SIZE//num_workers
        self.NUM_WORKERS = num_workers
        self.MAX_CYCLES = env_params['max_cycles']
        self.N_AGENTS = 2
        self.ACTION_SPACE = 1
        self.max_grad_norm = 10.0
        ray.init(num_cpus = self.NUM_WORKERS, _node_ip_address="0.0.0.0")

        self.envs = [Pong_Worker.remote(self.BATCH_SPLIT_SIZE, env_params, time_penalty=0.0) for _ in range(self.NUM_WORKERS)]
        #self.envs = [Piston_Worker(self.BATCH_SPLIT_SIZE, env_params, time_penalty=7e-3) for _ in range(self.NUM_WORKERS)]

    def map_actions(self, action_tens, num_left_per_worker):
        actions = []
        start_ix = 0
        #stuff = []
        for worker in range(0, self.NUM_WORKERS):
            num_left_worker = num_left_per_worker[worker]
            end_ix = start_ix + num_left_worker
            #stuff.append([start_ix, end_ix])
            actions_per_worker = action_tens[start_ix:end_ix]
            actions.append(self.action_to_dict(actions_per_worker))
            start_ix = end_ix
        #print('\t', stuff)
        return actions

    def action_to_dict(self, action_tens):
        num_batches, num_agents = action_tens.shape
        results = []
        for i in range(0, num_batches):
            results.append({ ('paddle_%d' % agent_ix): action_tens[i, agent_ix].item() for agent_ix in range(num_agents)})
        return results

    def get_render_run(self, experiment_dir):
        device=torch.device("cpu")
        print('**Using: cpu for inference')
        print('**Just getting a rendering!')
        self.encoder = Encoder(device)
        model = Combined_MOAPong_Helper(device)

        print('Evaluating: ', experiment_dir)
        with open(os.path.join('..','..', 'experiments','final_models', 'pong', experiment_dir, 'combined_model.pt'), 'rb') as f:
            d = torch.load(f, map_location=device)
        model.load_state_dicts(d['policy'])

        storage, rendering = self.train_epoch(model, device, render=True)
        epoch_returns = np.zeros((1, self.N_AGENTS, 2, self.BATCH_SIZE))
        team_total = 0
        for agent_ix in range(self.N_AGENTS):
            agent_loss, mean_rewards_over_batch, length_iteration_over_batch = storage[agent_ix].compute_loss(device)
            epoch_returns[0, agent_ix, 0] = mean_rewards_over_batch
            epoch_returns[0, agent_ix, 1] = length_iteration_over_batch
            team_total += mean_rewards_over_batch.mean().item()
            print('**Agent %s Return over Batches: %s' % (agent_ix, mean_rewards_over_batch.mean()))
            print('**Agent %s Survival Length over Batches: %s' % (agent_ix, length_iteration_over_batch.mean()))

        if input('wanna save? y or n: ') == 'y':
            np.save(os.path.join('..','..', 'videos', 'pong', '%s.npy' % experiment_dir), rendering)
        else:
            pass

    def eval(self, experiment_dir):
        device=torch.device("cpu")
        print('**Using: cpu for inference')
        print('**MOA')
        self.encoder = Encoder(device)
        model = Combined_MOAPong_Helper(device)

        print('Evaluating: ', experiment_dir)
        with open(os.path.join('..','..', 'experiments','final_models', 'pong', experiment_dir, 'combined_model.pt'), 'rb') as f:
            d = torch.load(f, map_location=device)
        model.load_state_dicts(d['policy'])

        trials = 100
        analytics = None
        for trial in range(0, trials):
            print('Trial %s/%s' % (trial, trials))
            storage = self.train_epoch(model, device)
            print('**Concluded Epoch Exploration')

            epoch_returns = np.zeros((1, self.N_AGENTS, 2, self.BATCH_SIZE))
            team_total = 0
            for agent_ix in range(self.N_AGENTS):
                agent_loss, mean_rewards_over_batch, length_iteration_over_batch = storage[agent_ix].compute_loss(device, gamma=0.95)
                epoch_returns[0, agent_ix, 0] = mean_rewards_over_batch
                epoch_returns[0, agent_ix, 1] = length_iteration_over_batch
                agent_loss.backward(retain_graph=True)
                team_total += mean_rewards_over_batch.mean().item()
                print('**Agent %s Return over Batches: %s' % (agent_ix, mean_rewards_over_batch.mean()))
                print('**Agent %s Survival Length over Batches: %s' % (agent_ix, length_iteration_over_batch.mean()))
            if analytics is None:
                analytics = epoch_returns
            else:
                analytics = np.append(analytics, epoch_returns, axis=0)
        np.save(os.path.join('..','..', 'eval_data', 'pong', '%s.npy' % experiment_dir), analytics)
        ray.shutdown()

    def train_loop(self, save_path, transfer_experiment=None):
        if torch.cuda.is_available():
            device=torch.device("cuda")
            torch.cuda.empty_cache()
            print('**Using:  for inference', torch.cuda.get_device_name(device))
            num_gpus = 1
        else:
            device=torch.device("cpu")
            num_gpus=0
            print('**Using: cpu for inference')

        print('**Doing MOA')
        self.encoder = Encoder(device)
        model = Combined_MOAPong_Helper(device)
        optimizers = [optim.Adam(param, lr=4e-4) for param in model.parameters()]

        if transfer_experiment is not None:
            print('Transferring from: ', transfer_experiment)
            with open(os.path.join('experiments', 'pong', transfer_experiment, 'combined_model.pt'), 'rb') as f:
                d = torch.load(f, map_location=device)
            model.load_state_dicts(d['policy'])
            for ix in range(0, self.N_AGENTS):
                optimizers[ix].load_state_dict(d['optimizer'][ix])

        epochs = 3000
        analytics = None
        best_performance = float('-inf')
        for epoch in range(0, epochs):
            print('Epoch %s/%s' % (epoch, epochs))
            storage = self.train_epoch(model, device)
            print('**Concluded Epoch Exploration')

            for agent_ix in range(0, self.N_AGENTS):
                optimizers[agent_ix].zero_grad()

            epoch_returns = np.zeros((1, self.N_AGENTS, 2, self.BATCH_SIZE))
            team_total = 0
            for agent_ix in range(self.N_AGENTS):
                agent_loss, mean_rewards_over_batch, length_iteration_over_batch = storage[agent_ix].compute_loss(device, gamma=0.95)
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

            if save_path is not None:
                self.save_checkpoint(model, optimizers, analytics, save_path)
                if team_total > best_performance:
                    best_performance = team_total
                    self.save_checkpoint(model, optimizers, analytics, os.path.join(save_path, 'best'))
        ray.shutdown()

    def save_checkpoint(self, model, optimizer, analytics, path):
        torch.save({
            'policy': model.state_dicts(),
            'optimizer': [op.state_dict() for op in optimizer]
        }, os.path.join(path, 'combined_model.pt'))

        np.save(os.path.join(path, 'data.npy'), analytics)

    def map_adj_matrix(self, adj_matrices, num_left_per_worker):
        total_left = sum(num_left_per_worker)
        giant_adj_matrix = np.zeros((total_left, self.N_AGENTS, self.N_AGENTS), dtype=bool)
        start_ix = 0
        for worker in range(0, self.NUM_WORKERS):
            num_left_worker = num_left_per_worker[worker]
            end_ix = start_ix + num_left_worker
            adj_matrix = adj_matrices[worker]
            assert len(adj_matrix) == num_left_worker, 'Error with adj_matrix from worker %s, length of adj_matrix=%s, but num_left = %s' % (worker, len(adj_matrix), num_left_per_worker)
            giant_adj_matrix[start_ix:end_ix] = adj_matrix
            start_ix = end_ix
        return torch.from_numpy(giant_adj_matrix)

    def train_epoch(self, model, device, render=False):
        observations = ray.get([env.reset.remote() for env in self.envs])
        storage = [Batch_Storage(batch_size=self.BATCH_SIZE, workers=self.NUM_WORKERS) for _ in range(self.N_AGENTS)]
        previous_actions_all_workers = [torch.zeros(self.BATCH_SPLIT_SIZE, self.N_AGENTS, 3).to(device) for _ in range(self.NUM_WORKERS)]

        self.encoder = Encoder(device)
        moa_loss = nn.CrossEntropyLoss(reduction='none')
        output_render_obj = None
        for step in range(0, self.MAX_CYCLES):
            #print(step)
            num_left_per_worker = [len(obs) for obs in observations]
            assert len(num_left_per_worker) == self.NUM_WORKERS, 'Error with retrieving observations'
            #print('\t', step, num_left_per_worker)
            if sum(num_left_per_worker) == 0:
                break

            joined_observations = np.concatenate(observations, axis=0)
            observations = [joined_observations[:, agent_ix, ...].copy() for agent_ix in range(self.N_AGENTS)]
            encodings = [self.encoder(obs) for obs in observations]

            join_previous_actions = torch.cat(previous_actions_all_workers, dim=0)
            assert len(join_previous_actions) == len(joined_observations), 'error here'

            output_action_dist, state_vals, actions_of_neighbors = model(encodings, join_previous_actions)

            reformatted_moa_action_predictions = torch.cat([dist.unsqueeze(1) for dist in actions_of_neighbors], dim=1)# batch x agents x 3
            reformatted_action_dist = torch.cat([dist.unsqueeze(1) for dist in output_action_dist], dim=1) # batch x agents x 3

            torch_dist = Categorical(probs=reformatted_action_dist)
            batched_actions = torch_dist.sample()
            batched_log_probs = torch_dist.log_prob(batched_actions).unsqueeze(dim=-1)
            actions_per_worker = self.map_actions(batched_actions, num_left_per_worker)

            outputs = ray.get([self.envs[proc_ix].step.remote(actions_per_worker[proc_ix]) for proc_ix in range(0, self.NUM_WORKERS)])

            moa_input = reformatted_moa_action_predictions.permute((0, 2, 1)) #batchx3(classes)x2(agents)
            moa_target = batched_actions.clone().detach()
            moa_target[:, [0,1]] = moa_target[:, [1,0]] #batchx2(agents), actions of other agents

            cross_entropy_loss = moa_loss(moa_input, moa_target) #batchx2(agents), loss per agent across batches

            observations = [out[0] for out in outputs]
            start_ix = 0
            for worker_ix in range(0, self.NUM_WORKERS):
                rewards = outputs[worker_ix][1].copy()
                dones = outputs[worker_ix][2].copy()
                end_ix = start_ix + num_left_per_worker[worker_ix]
                for agent_ix in range(0, self.N_AGENTS):
                    storage[agent_ix].add_to_batch(worker_ix, dones,
                                                   batched_log_probs[start_ix:end_ix, agent_ix],
                                                   state_vals[agent_ix][start_ix:end_ix],
                                                   rewards[:, agent_ix],
                                                   cross_entropy_loss[start_ix:end_ix, agent_ix])
                start_ix = end_ix

            dones_next = [out[3] for out in outputs]
            infos = [out[4] for out in outputs]

            start_ix = 0
            for worker_ix in range(0, self.NUM_WORKERS):
                end_ix = start_ix + num_left_per_worker[worker_ix]
                info_at_worker = infos[worker_ix]
                prev_latents = previous_actions_all_workers[worker_ix]
                latent_policies = [output_action_dist[1][start_ix:end_ix], output_action_dist[0][start_ix:end_ix]]
                instant_dones_next = torch.tensor(dones_next[worker_ix].copy(), dtype=torch.bool)

                assert len(info_at_worker) == len(prev_latents) == len(instant_dones_next) == end_ix - start_ix, 'Error here'
                for batch_ix in range(len(info_at_worker)):
                    if info_at_worker[batch_ix][0]: # paddle 0 hit, save its latent_policy and reset paddle 1's latent policy
                        #print('pong_0 hit!')
                        prev_latents[batch_ix][0] = latent_policies[0][batch_ix].detach().clone()
                        prev_latents[batch_ix][1] = torch.zeros((3))
                    if info_at_worker[batch_ix][1]:
                        #print('pong_1 hit!')
                        prev_latents[batch_ix][1] = latent_policies[1][batch_ix].detach().clone()
                        prev_latents[batch_ix][0] = torch.zeros((3))

                previous_actions_all_workers[worker_ix] = prev_latents[~instant_dones_next, ...]
                start_ix = end_ix
            if render:
                temp_rendering = ray.get([self.envs[proc_ix].render.remote() for proc_ix in range(0, self.NUM_WORKERS)])
                np_rendering = np.concatenate([thing.copy() for thing in temp_rendering], axis=0)
                if output_render_obj is None:
                    output_render_obj = np_rendering
                else:
                    output_render_obj = np.append(output_render_obj, np_rendering, axis=0)

        if render:
            return storage, output_render_obj
        else:
            return storage

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', type=int, required=True, help="Batch Size")
    parser.add_argument('-workers', type=int, required=True, help="Number of CPUS")
    parser.add_argument('-transfer_experiment', type=str, required=False, default=None, help='Continue learning from some previous experiment?')
    parser.add_argument('-eval', action='store_true', required=False, default=False, help="Eval Mode?")

    args = parser.parse_args()

    #if args.consensus and args.k > 0:
    #    raise Exception('Cant do consensus update and K-Level Communication at the same time!')

    if not args.eval:
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        path = os.path.join('experiments', 'pong', experiment_name)
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'best'))

        with open(os.path.join(path, 'params.txt'), 'w') as f:
            f.write('Args: batch %s, workers %s, MOA' % (
                args.batch, args.workers))
        run = Runner(args.batch, num_workers=args.workers)
        run.train_loop(path, transfer_experiment=args.transfer_experiment)
    else:
        if args.transfer_experiment is None:
            raise Exception('In order to eval, you need to provide a source directory')
        run = Runner(args.batch, num_workers=args.workers)
        run.eval(experiment_dir=args.transfer_experiment)
