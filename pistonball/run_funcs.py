from batch_piston_env import PistonEnv
from policy_piston import PistonPolicy
import torch
import torch.optim as optim
import numpy as np
import os


def test_piston_with_hyperparams(hyper_params, verbose):
    if torch.cuda.is_available():
        device=torch.device("c uda:0")
        print('**Using: ', torch.cuda.get_device_name(device))
    else:
        device=torch.device("cpu")
        print('**Using: cpu')
        
    action_space = 3
    encoding_size = hyper_params['encoding_size']
    policy_latent_size = hyper_params['policy_latent_size']
    lr = hyper_params['lr']
    epochs = hyper_params['epochs']
    n_agents = hyper_params['n_agents']
    max_cycles = hyper_params['max_cycles']
    max_grad_norm = hyper_params['max_grad_norm']
    communicate = hyper_params['communicate']
    time_penalty = hyper_params['time_penalty']
    early_reward_benefit = hyper_params['early_reward_benefit']
    batch_size = hyper_params['batch_size']
    if 'adv' in hyper_params.keys():
        adv_type = hyper_params['adv']
    else:
        adv_type = 'normal'

    print('*With %s Agents' % (n_agents))
    print('**Using %s-Agent with Batch Size: %s' % (n_agents, batch_size))
    if 'consensus_update' in hyper_params.keys():
        consensus_update = hyper_params['consensus_update']
        if consensus_update and communicate:
            raise Exception('Shldnt be running InfoPG and ConsensusUpdate at the same time')
    else:
        consensus_update = False
        
    if 'k-levels' in hyper_params.keys():
        k_levels = hyper_params['k-levels']

        if k_levels == 0 and communicate:
            raise Exception('In order to communicate k should be at least 1')

        if k_levels > 0 and consensus_update:
            raise Exception('Shldnt be setting k-levels and ConsensusUpdate at the same time')
    else:
        k_levels = 1
        
    env_params = {
        'n_pistons': n_agents, 'local_ratio': 1.0, 'time_penalty': 0.0, 'continuous': False,
        'random_drop': True, 'random_rotate': True, 'ball_mass': 0.75, 'ball_friction': 0.3,
        'ball_elasticity': 1.5, 'max_cycles': max_cycles
    }
    user_params = {
        'device': device,
        'epochs': epochs,
        'verbose': verbose,
        'communicate': communicate,
        'max_grad_norm': max_grad_norm,
        'time_penalty': time_penalty,
        'early_reward_benefit': early_reward_benefit,
        'consensus_update': consensus_update,
        'k-levels': k_levels,
    }

    env = PistonEnv(batch_size, env_params)
    if hyper_params['transfer_experiment'] is not None:
        policies, optimizers = create_policies_from_experiment(hyper_params, device)
    else:
        policies = {agent: PistonPolicy(encoding_size, policy_latent_size, action_space, device, adv_type) for agent in env.get_agent_names()}
        optimizers = {agent: optim.Adam(policies[agent].parameters(), lr) for agent in env.get_agent_names()}
    if hyper_params['scheduler'] is not None:
        if hyper_params['scheduler']['type'] == 'MultiStepLR':
            print('*Using MultiStepLR with milestones at: %s and gamma factor: %s' % (hyper_params['scheduler']['params']['milestones'],
                                                                                      hyper_params['scheduler']['params']['gamma']))
            schedulers = {agent: optim.lr_scheduler.MultiStepLR(optimizer=optimizers[agent], **hyper_params['scheduler']['params']) for agent in env.get_agent_names()}
        else:
            raise Exception("%s isnt supported yet" % (hyper_params['scheduler']['type']))
    else:
        schedulers = None
    policies, optimizers, summary_stats = env.loop(user_params, policies, optimizers, schedulers)
    return summary_stats, policies, optimizers

def create_policies_from_experiment(hyper_params, device):
    encoding_size = hyper_params['encoding_size']
    policy_latent_size = hyper_params['policy_latent_size']
    experiment_name = hyper_params['transfer_experiment']['name']
    piston_order = hyper_params['transfer_experiment']['order']
    if len(piston_order) != hyper_params['n_agents']:
        raise Exception('Must put in an ordering that is equal to the number of required agents')
    lr = hyper_params['lr']
    files = os.listdir(os.path.join('experiments', 'pistonball', experiment_name))
    files = list(filter(lambda x: '.pt' in x, files))
    if len(set(files)) < len(set(piston_order)):
        raise Exception("Agents aren't the ones from the transfer experiments")
    n_agents = len(files)
    policies = {}
    optimizers = {}
    action_space = 3
    for i in range(0, len(piston_order)):
        agent_name = piston_order[i]
        data = torch.load(os.path.join('experiments', 'pistonball', experiment_name, '%s.pt' % (agent_name)), device)
        print('giving %s, %s experimental policy' % ('piston_%s' % (i), agent_name))
        policies['piston_%s' % (i)] = PistonPolicy(encoding_size, policy_latent_size, action_space, device, 'normal',
                                                   model_state_dict=data['policy'])
        optimizers['piston_%s' % (i)] = optim.Adam(policies['piston_%s' % (i)].parameters(), lr)
        optimizers['piston_%s' % (i)].load_state_dict(data['optimizer'])
    return policies, optimizers