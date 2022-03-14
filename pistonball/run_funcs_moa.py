from batch_piston_env import PistonEnv_MOA
from policy_piston import MOAPolicy
import torch
import os
import torch.optim as optim

def test_piston_with_hyperparams_moa(hyper_params, verbose):
    if 'moa' in hyper_params:
        if not hyper_params['moa']:
            raise Exception('In order to use moa, must be set to true')
    else:
        raise Exception('Must set moa key to true')

    if torch.cuda.is_available():
        device=torch.device("cuda:0")
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
    time_penalty = hyper_params['time_penalty']
    early_reward_benefit = hyper_params['early_reward_benefit']
    batch_size = hyper_params['batch_size']

    print('*With %s Agents' % (n_agents))
    print('**Using %s-Agent with Batch Size: %s' % (n_agents, batch_size))

    env_params = {
        'n_pistons': n_agents, 'local_ratio': 1.0, 'time_penalty': 0.0, 'continuous': False,
        'random_drop': True, 'random_rotate': True, 'ball_mass': 0.75, 'ball_friction': 0.3,
        'ball_elasticity': 1.5, 'max_cycles': max_cycles
    }
    user_params = {
        'device': device,
        'epochs': epochs,
        'verbose': verbose,
        'max_grad_norm': max_grad_norm,
        'time_penalty': time_penalty,
        'early_reward_benefit': early_reward_benefit,
    }

    env = PistonEnv_MOA(batch_size, env_params)
    if hyper_params['transfer_experiment'] is not None:
        policies, optimizers = create_policies_from_experiment_moa(hyper_params, device)
    else:
        policies = {agent: MOAPolicy(encoding_size, policy_latent_size, action_space, device) for agent in env.get_agent_names()}
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

def create_policies_from_experiment_moa(hyper_params, device):
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
        policies['piston_%s' % (i)] = MOAPolicy(encoding_size, policy_latent_size, action_space, device,
                                                   model_state_dict=data['policy'])
        optimizers['piston_%s' % (i)] = optim.Adam(policies['piston_%s' % (i)].parameters(), lr)
        optimizers['piston_%s' % (i)].load_state_dict(data['optimizer'])
    return policies, optimizers