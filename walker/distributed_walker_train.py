import torch
import numpy as np
import ray
import torch.optim as optim
import os
from distributed_walker_policy import Combined_Walker_Helper
from distributed_walker_env import Walker_Worker
from storage import Batch_Storage
from torch.distributions import Normal
import datetime

class Runner:
    def __init__(self, batch_size, num_workers = 4):
        env_params = {
            'n_walkers': 2, 'position_noise': 1e-4, 'angle_noise': 1e-4, 'local_ratio': 1.0,
            'forward_reward': 0.0, 'terminate_reward':-100.0, 'fall_reward':-10.0, 'terminate_on_fall': True,
            'remove_on_fall': True, 'max_cycles': 500, 'use_package': True
        }
        assert batch_size % num_workers == 0
        self.BATCH_SIZE = batch_size
        self.BATCH_SPLIT_SIZE = self.BATCH_SIZE//num_workers
        self.NUM_WORKERS = num_workers
        self.MAX_CYCLES = env_params['max_cycles']
        self.N_AGENTS = env_params['n_walkers']
        self.ACTION_SPACE = 4
        self.max_grad_norm = 5.0
        ray.init(num_cpus = self.NUM_WORKERS, _node_ip_address="0.0.0.0")
        self.envs = [Walker_Worker.remote(self.BATCH_SPLIT_SIZE, env_params) for _ in range(self.NUM_WORKERS)]
        #self.envs = [Walker_Worker(self.BATCH_SPLIT_SIZE, env_params, time_penalty=7e-3) for _ in range(self.NUM_WORKERS)]

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
        num_batches, num_agents, action_space = action_tens.shape
        results = []
        for i in range(0, num_batches):
            results.append({ ('walker_%d' % agent_ix): action_tens[i, agent_ix].to('cpu').numpy() for agent_ix in range(num_agents)})
        return results

    def get_render_run(self, experiment_dir, k_levels=1):
        device=torch.device("cpu")
        print('**Using: cpu for inference')
        print('**Just getting a rendering!')
        model = Combined_Walker_Helper(device, k_levels)

        print('Evaluating: ', experiment_dir)
        with open(os.path.join('..','..', 'experiments','final_models', 'walker', experiment_dir, 'combined_model.pt'), 'rb') as f:
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
            np.save(os.path.join('..','..', 'videos', 'walker', '%s.npy' % experiment_dir), rendering)
        else:
            pass

    def eval(self, experiment_dir, k_levels=1, adv='normal', use_critic=False, consensus=False):
        device=torch.device("cpu")
        print('**Using: cpu for inference')
        print('**Using k-levels: %s and advantage_func: %s and critic: %s' % (k_levels, adv, use_critic))
        model = Combined_Walker_Helper(device, k_levels=k_levels)

        print('Evaluating: ', experiment_dir)
        with open(os.path.join('..','..', 'experiments','final_models', 'walker', experiment_dir, 'combined_model.pt'), 'rb') as f:
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
        np.save(os.path.join('..', '..', 'eval_data', 'walker', '%s.npy' % experiment_dir), analytics)
        ray.shutdown()

    def train_loop(self, save_path, k_levels=1, adv='normal', use_critic=False, transfer_experiment=None, use_mlp=False):
        if torch.cuda.is_available():
            device=torch.device("cuda")
            torch.cuda.empty_cache()
            print('**Using:  for inference', torch.cuda.get_device_name(device))
            num_gpus = 1
        else:
            device=torch.device("cpu")
            num_gpus=0
            print('**Using: cpu for inference')

        print('**Using batch size: %s' % self.BATCH_SIZE)
        print('**Using k-levels: %s and advantage_func: %s and critic: %s' % (k_levels, adv, use_critic))
        model = Combined_Walker_Helper(device, k_levels=k_levels, n_agents=self.N_AGENTS, use_mlp=use_mlp)
        model.to(device)
        if not use_mlp:
            my_list = ['to_means.0.weight', 'to_means.0.bias', 'recurr_policy.weight_ih_l0', 'recurr_policy.weight_hh_l0',
                       'recurr_policy.bias_ih_l0', 'recurr_policy.bias_hh_l0', 'v_net.0.weight', 'v_net.0.bias']
        else:
            my_list = ['to_means.0.weight', 'to_means.0.bias', 'recurr_policy.0.weight', 'recurr_policy.0.bias',
                       'recurr_policy.2.weight', 'recurr_policy.2.bias', 'v_net.0.weight', 'v_net.0.bias']
        optimizers = []
        for agent in model.agents:
            trainable_params = []
            base_params = []
            for param_name, param in agent.named_parameters():
                if param_name in my_list:
                    trainable_params.append(param)
                else:
                    base_params.append(param)
            optimizers.append(
                optim.Adam(
                    [{'params': base_params}, {'params': trainable_params, 'lr': 1e-4}], lr=3e-6
                )
            )
        optimizers = [optim.Adam(param, lr=4e-4) for param in model.parameters()]

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
            encodings = [torch.tensor(obs, dtype=torch.float32).to(device) for obs in observations]
            output_action_dist, state_vals = model(encodings, None)

            output = torch.cat([dist.unsqueeze(1) for dist in output_action_dist], dim=1) # batch x agents x 8

            output_means = output[:,:, 0:4]
            output_log_std = torch.full(output_means.shape, fill_value=-0.91).to(device)
            torch_dist = Normal(loc=output_means, scale=torch.exp(output_log_std))
            batch_torch_actions = torch_dist.sample().clamp(min=-1, max=1)
            batched_log_probs = torch_dist.log_prob(batch_torch_actions)
            batched_entropies = torch_dist.entropy()

            actions_per_worker = self.map_actions(batch_torch_actions, num_left_per_worker)
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
                                                   rewards[:, agent_ix],
                                                   batched_entropies[start_ix:end_ix, agent_ix])
                start_ix = end_ix
            if render:
                print(step)
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

    # python distributed_walker_train.py -batch 32 -workers 8 -k 0 -adv normal
    # python distributed_walker_train.py -batch 32 -workers 8 -k 0 -adv normal -consensus

    # python distributed_walker_train.py -batch 32 -workers 8 -k 1 -adv info -mlp
    # python distributed_walker_train.py -batch 32 -workers 8 -k 1 -adv info

    # python distributed_walker_train.py -batch 32 -workers 8 -k 1 -adv normal -mlp << add the entropy back
    # python distributed_walker_train.py -batch 32 -workers 8 -k 1 -adv normal << add the entropy back
    parser = argparse.ArgumentParser()

    parser.add_argument('-batch', type=int, required=True, help="Batch Size")
    parser.add_argument('-workers', type=int, required=True, help="Number of CPUS")
    parser.add_argument('-k', type=int, required=True, default=1, help="Number of K Levels")
    parser.add_argument('-adv', type=str, required=True, default='normal', help="Use Normal Advantage or Info Adv, options = 'normal' or 'info'")

    parser.add_argument('-eval', action='store_true', required=False, default=False, help="Eval Mode?")
    parser.add_argument('-mlp', action='store_true', required=False, default=False, help="Use MLP for Recurrent Layer?")
    parser.add_argument('-transfer_experiment', type=str, required=False, default=None, help='Continue learning from some previous experiment?')
    parser.add_argument('-critic', action='store_true', required=False, default=False, help="Use Critic in Pong?")
    parser.add_argument('-consensus', action='store_true', required=False, default=False, help="Using Consensus Update")
    args = parser.parse_args()

    if args.consensus and args.k > 0:
        raise Exception('Cant do consensus update and K-Level Communication at the same time!')

    workers = args.workers
    run = Runner(args.batch, num_workers=workers)
    if not args.eval:
        experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
        path = os.path.join('experiments', 'walker', experiment_name)
        os.makedirs(path)
        os.makedirs(os.path.join(path, 'best'))

        with open(os.path.join(path, 'params.txt'), 'w') as f:
            f.write('Args: batch %s, workers %s, k-levels %s, adv %s, critic %s, consensus %s' % (
                args.batch, args.workers, args.k, args.adv, args.critic, args.consensus
            ))
        run.train_loop(path, k_levels = args.k, adv=args.adv, use_critic=args.critic,
                       transfer_experiment=args.transfer_experiment, use_mlp=args.mlp)
    else:
        if args.transfer_experiment is None:
            raise Exception('In order to eval, you need to provide a source directory')
        run.eval(experiment_dir=args.transfer_experiment, k_levels=args.k, adv=args.adv, use_critic=args.critic, consensus=args.consensus)
