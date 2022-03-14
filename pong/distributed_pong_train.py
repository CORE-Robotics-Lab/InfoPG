from distributed_pong_env import Pong_Worker
from distributed_pong_policy import Combined_Pong_Helper
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


class Runner:
    def __init__(self, batch_size, num_workers = 4):
        env_params = {
            'ball_speed': 15, 'left_paddle_speed': 13, 'right_paddle_speed': 13,
            'cake_paddle': False, 'max_cycles': 500, 'bounce_randomness': False
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

    def get_render_run(self, experiment_dir, k_levels=1):
        device=torch.device("cpu")
        print('**Using: cpu for inference')
        print('**Just getting a rendering!')
        self.encoder = Encoder(device)
        model = Combined_Pong_Helper(device, k_levels=k_levels)

        print('Evaluating: ', experiment_dir)
        with open(os.path.join('..', 'experiments','final_models', 'pong', experiment_dir, 'combined_model.pt'), 'rb') as f:
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
            np.save(os.path.join('..', 'videos', 'pong', '%s.npy' % experiment_dir), rendering)
        else:
            pass


    def eval(self, experiment_dir, k_levels=1, adv='normal', use_critic=False, consensus=False):
        device=torch.device("cpu")
        print('**Using: cpu for inference')
        print('**Using k-levels: %s and advantage_func: %s and critic: %s' % (k_levels, adv, use_critic))
        self.encoder = Encoder(device)
        model = Combined_Pong_Helper(device, k_levels=k_levels)

        print('Evaluating: ', experiment_dir)
        with open(os.path.join('..', 'experiments','final_models', 'pong', experiment_dir, 'combined_model.pt'), 'rb') as f:
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
                agent_loss, mean_rewards_over_batch, length_iteration_over_batch = storage[agent_ix].compute_loss(device, adv_opt=adv, use_critic=use_critic, gamma=0.95)
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
        np.save(os.path.join('..', 'eval_data', 'pong', '%s.npy' % experiment_dir), analytics)
        ray.shutdown()

    def train_loop(self, save_path, k_levels=1, adv='normal', use_critic=False, transfer_experiment=None, consensus=False):
        if torch.cuda.is_available():
            device=torch.device("cuda")
            torch.cuda.empty_cache()
            print('**Using:  for inference', torch.cuda.get_device_name(device))
            num_gpus = 1
        else:
            device=torch.device("cpu")
            num_gpus=0
            print('**Using: cpu for inference')

        print('**Using k-levels: %s and advantage_func: %s and critic: %s' % (k_levels, adv, use_critic))
        self.encoder = Encoder(device)
        model = Combined_Pong_Helper(device, k_levels=k_levels)
        optimizers = [optim.Adam(param, lr=4e-4) for param in model.parameters()]

        if transfer_experiment is not None:
            print('Transferring from: ', transfer_experiment)
            with open(os.path.join('experiments', 'pong', transfer_experiment, 'combined_model.pt'), 'rb') as f:
                d = torch.load(f, map_location=device)
            model.load_state_dicts(d['policy'])
            for ix in range(0, self.N_AGENTS):
                optimizers[ix].load_state_dict(d['optimizer'][ix])

        epochs = 1000
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
                agent_loss, mean_rewards_over_batch, length_iteration_over_batch = storage[agent_ix].compute_loss(device, adv_opt=adv, use_critic=use_critic, gamma=0.95)
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
        ray.shutdown()

    def save_checkpoint(self, model, optimizer, analytics, path):
        torch.save({
            'policy': model.state_dicts(),
            'optimizer': [op.state_dict() for op in optimizer]
        }, os.path.join(path, 'combined_model.pt'))

        np.save(os.path.join(path, 'data.npy'), analytics)

    def train_epoch(self, model, device, render=False):
        observations = ray.get([env.reset.remote() for env in self.envs])
        storage = [Batch_Storage(batch_size=self.BATCH_SIZE, workers=self.NUM_WORKERS) for _ in range(self.N_AGENTS)]
        latents_all_workers = [torch.zeros(self.BATCH_SPLIT_SIZE, 2, 30).to(device) for _ in range(self.NUM_WORKERS)]

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
            adj_matrices = ray.get([env.get_adj_matrix.remote() for env in self.envs])

            join_adj_matrices = self.map_adj_matrix(adj_matrices, num_left_per_worker)
            join_adj_matrices = join_adj_matrices.to(device)

            join_latent_vectors = torch.cat(latents_all_workers, dim=0)
            assert len(join_latent_vectors) == len(joined_observations), 'error here'

            output_action_dist, state_vals, policy_latent = model(encodings, join_latent_vectors)

            reformatted_action_dist = torch.cat([dist.unsqueeze(1) for dist in output_action_dist], dim=1) # batch x agents x 3
            torch_dist = Categorical(probs=reformatted_action_dist)
            batched_actions = torch_dist.sample()
            batched_log_probs = torch_dist.log_prob(batched_actions).unsqueeze(dim=-1)
            #batched_entropies = torch_dist.entropy()
            actions_per_worker = self.map_actions(batched_actions, num_left_per_worker)

            outputs = ray.get([self.envs[proc_ix].step.remote(actions_per_worker[proc_ix]) for proc_ix in range(0, self.NUM_WORKERS)])
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
                                                   rewards[:, agent_ix])
                start_ix = end_ix

            dones_next = [out[3] for out in outputs]
            infos = [out[4] for out in outputs]

            start_ix = 0
            for worker_ix in range(0, self.NUM_WORKERS):
                end_ix = start_ix + num_left_per_worker[worker_ix]
                info_at_worker = infos[worker_ix]
                prev_latents = latents_all_workers[worker_ix]
                latent_policies = [policy_latent[0][start_ix:end_ix], policy_latent[1][start_ix:end_ix]]
                instant_dones_next = torch.tensor(dones_next[worker_ix].copy(), dtype=torch.bool)

                assert len(info_at_worker) == len(prev_latents) == len(instant_dones_next) == end_ix - start_ix, 'Error here'
                for batch_ix in range(len(info_at_worker)):
                    if info_at_worker[batch_ix][0]: # paddle 0 hit, save its latent_policy and reset paddle 1's latent policy
                        #print('pong_0 hit!')
                        prev_latents[batch_ix][0] = latent_policies[0][batch_ix]
                        prev_latents[batch_ix][1] = torch.zeros((30))
                    if info_at_worker[batch_ix][1]:
                        #print('pong_1 hit!')
                        prev_latents[batch_ix][1] = latent_policies[1][batch_ix]
                        prev_latents[batch_ix][0] = torch.zeros((30))

                latents_all_workers[worker_ix] = prev_latents[~instant_dones_next, ...]
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

    # nohup python distributed_pong_train.py -batch 16 -workers 8 -k 1 -adv info -transfer_experiment 2021-05-23\ 01_00_14 >> k1_info_no_critic.log &
    # nohup python distributed_pong_train.py -batch 16 -workers 8 -k 1 -adv info -critic -transfer_experiment 2021-05-23\ 01_00_32 >> k1_info_critic.log &
    # nohup python distributed_pong_train.py -batch 16 -workers 8 -k 0 -adv normal -transfer_experiment 2021-05-23\ 01_00_48 >> nc_normal_adv.log &
    # nohup python distributed_pong_train.py -batch 16 -workers 8 -k 1 -adv info -transfer_experiment 2021-05-23\ 01_50_56 >> k1_info_no_clamp.log &

    # nohup python distributed_pong_train.py -batch 16 -workers 8 -k 0 -adv normal -consensus -transfer_experiment 2021-05-24\ 02_09_42 >> consensus_extended.log &
    # nohup python distributed_pong_train.py -batch 16 -workers 8 -k 0 -adv normal -transfer_experiment 2021-05-23\ 14_47_37 >> nc_normal_adv.log &
    # nohup python distributed_pong_train.py -batch 16 -workers 8 -k 1 -adv info -critic -transfer_experiment 2021-05-23\ 14_47_10 >> k1_info_critic.log &
    # nohup python distributed_pong_train.py -batch 16 -workers 8 -k 2 -adv info -critic -transfer_experiment 2021-05-24\ 02_09_52>> k2_extended.log &
    # nohup python distributed_pong_train.py -batch 16 -workers 8 -k 3 -adv info -critic -transfer_experiment 2021-05-24\ 02_10_09 >> k3_extended.log &

    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', type=int, required=True, help="Batch Size")
    parser.add_argument('-workers', type=int, required=True, help="Number of CPUS")
    parser.add_argument('-k', type=int, required=True, default=1, help="Number of K Levels")
    parser.add_argument('-adv', type=str, required=True, default='normal', help="Use Normal Advantage or Info Adv, options = 'normal' or 'info'")
    parser.add_argument('-transfer_experiment', type=str, required=False, default=None, help='Continue learning from some previous experiment?')

    parser.add_argument('-eval', action='store_true', required=False, default=False, help="Eval Mode?")
    parser.add_argument('-critic', action='store_true', required=False, default=False, help="Use Critic in Pong?")
    parser.add_argument('-consensus', action='store_true', required=False, default=False, help="Using Consensus Update")
    args = parser.parse_args()

    if args.consensus and args.k > 0:
        raise Exception('Cant do consensus update and K-Level Communication at the same time!')

    workers = args.workers

    run = Runner(args.batch, num_workers=workers)
    if not args.eval:
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        path = os.path.join('experiments', 'pong', experiment_name)
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'best'))

        with open(os.path.join(path, 'params.txt'), 'w') as f:
            f.write('Args: batch %s, workers %s, k-levels %s, adv %s, critic %s, consensus %s' % (
                args.batch, args.workers, args.k, args.adv, args.critic, args.consensus
            ))

        run.train_loop(path, k_levels = args.k, adv=args.adv, use_critic=args.critic, transfer_experiment=args.transfer_experiment, consensus=args.consensus)
    else:
        # python distributed_pong_train.py -eval -transfer_experiment k_1
        if args.transfer_experiment is None:
            raise Exception('In order to eval, you need to provide a source directory')
        run.eval(experiment_dir=args.transfer_experiment, k_levels=args.k, adv=args.adv, use_critic=args.critic, consensus=args.consensus)