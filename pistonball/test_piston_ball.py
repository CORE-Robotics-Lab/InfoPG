from run_funcs import *
from run_funcs_moa import *
import datetime
import pickle
import os
import torch

torch.backends.cudnn.deterministic = True

def train_a2c():
    hyper_params = {
        'encoding_size': 300,
        'policy_latent_size': 20,
        'lr': 0.001,
        'epochs': 1000,
        'n_agents': 5,
        'max_cycles': 200,
        'max_grad_norm': 0.75,
        'communicate': False,
        'consensus_update': False,
        'transfer_experiment': None,
        'time_penalty': 0.007,
        'early_reward_benefit': 0.25,
        'batch_size': 4,
        'k-levels': 0,
        'scheduler': None
    }

    data, policies, optimizers = test_piston_with_hyperparams(hyper_params, verbose=True)
    return data, policies, optimizers, hyper_params

def train_with_consensus_5():
    hyper_params = {
        'encoding_size': 300,
        'policy_latent_size': 20,
        'lr': 0.001,
        'epochs': 1000,
        'n_agents': 5,
        'max_cycles': 200,
        'max_grad_norm': 0.8,
        'transfer_experiment': None,
        'time_penalty': 7e-3,
        'early_reward_benefit': 0.25,
        'batch_size': 4,
        'scheduler': None,
        'communicate': False,
        'consensus_update': True,
        'k-levels': 0,
    }
    data, policies, optimizers = test_piston_with_hyperparams(hyper_params, verbose=True)
    return data, policies, optimizers, hyper_params

def train_adv_infopg_5(k_levels):
    hyper_params = {
        'encoding_size': 300,
        'policy_latent_size': 20,
        'lr': 0.001,
        'epochs': 1000,
        'n_agents': 5,
        'max_cycles': 200,
        'max_grad_norm': 0.75,
        'communicate': True,
        'transfer_experiment': None,
        'time_penalty': 0.007,
        'early_reward_benefit': 0.25,
        'batch_size': 4,
        'k-levels': k_levels,
        'scheduler': None,
        'adv': 'normal',
    }

    data, policies, optimizers = test_piston_with_hyperparams(hyper_params, verbose=True)
    return data, policies, optimizers, hyper_params


def train_infopg_5(k_levels):
    hyper_params = {
        'encoding_size': 300,
        'policy_latent_size': 20,
        'lr': 0.001,
        'epochs': 1000,
        'n_agents': 5,
        'max_cycles': 200,
        'max_grad_norm': 0.75,
        'communicate': True,
        'transfer_experiment': None,
        'time_penalty': 0.007,
        'early_reward_benefit': 0.85,
        'batch_size': 4,
        'k-levels': k_levels,
        'scheduler': None,
        'adv': 'clamped_q',
    }

    data, policies, optimizers = test_piston_with_hyperparams(hyper_params, verbose=True)

    return data, policies, optimizers, hyper_params

def train_moa_5():
    hyper_params = {
        'moa': True,
        'encoding_size': 300,
        'policy_latent_size': 20,
        'lr': 0.001,
        'epochs': 1000,
        'n_agents': 5,
        'max_cycles': 200,
        'max_grad_norm': 0.75,
        'transfer_experiment': None,
        'time_penalty': 0.007,
        'early_reward_benefit': 0.25,
        'batch_size': 4,
        'scheduler': None,
    }
    data, policies, optimizers = test_piston_with_hyperparams_moa(hyper_params, verbose=True)
    return data, policies, optimizers, hyper_params

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-method', type=str, required=True, help='Method to Run: (moa, infopg, infopg_adv, consensus, a2c)')
    parser.add_argument('-k', type=int, required=False, default=1, help='If using infopg or infopg_adv, specify a k?')
    args = parser.parse_args()

    if args.method == 'moa':
        data, policies, optimizers, hyper_params = train_moa_5()
    elif args.method == 'infopg':
        data, policies, optimizers, hyper_params = train_infopg_5(args.k)
    elif args.method == 'infopg_adv':
        data, policies, optimizers, hyper_params = train_adv_infopg_5(args.k)
    elif args.method == 'consensus':
        data, policies, optimizers, hyper_params = train_with_consensus_5()
    elif args.method == 'a2c':
        data, policies, optimizers, hyper_params = train_a2c()
    else:
        raise Exception(args.method + " isnt accepted")

    experiment_name = datetime.datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    path = os.path.join('experiments', 'pistonball', experiment_name)
    os.makedirs(path)

    with open('%s.pkl' % (os.path.join(path, 'data')), 'wb') as f:
        pickle.dump(data, f)
    with open('%s.pkl' % (os.path.join(path, 'hyper_params')), 'wb') as f:
        pickle.dump(hyper_params, f)
    for agent in policies.keys():
        torch.save({
            'policy': policies[agent].state_dict(),
            'optimizer': optimizers[agent].state_dict()
        }, os.path.join(path, agent+'.pt'))
