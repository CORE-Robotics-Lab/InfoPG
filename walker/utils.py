import torch.nn as nn
import torch
import os

def transfer_bipedal_to_multiwalker(agent: nn.Module, eval = False):
    with open(os.path.join('single_biped_weights', 'best_weight_2.pt'), 'rb') as f:
        state_dict = torch.load(f)
        new_state_dict = {}
        for k,v in state_dict.items():
            new_state_dict[k[k.find('.')+1:]] = torch.tensor(v)
        transfer_keys = ['down_sampler.0.weight', 'down_sampler.0.bias', 'down_sampler.2.weight', 'down_sampler.2.bias',
                         'policy.0.weight', 'policy.0.bias']
        if eval:
            transfer_keys += ['to_means.0.weight', 'to_means.0.bias', 'v_net.0.weight', 'v_net.0.bias']

        agent_state_dict = agent.state_dict()
        for k in transfer_keys:
            agent_state_dict[k] = new_state_dict[k].requires_grad_(True)
        agent.load_state_dict(agent_state_dict)

#if __name__ == '__main__':
#    from distributed_walker_policy import Combined_Walker_Helper
#    model = Combined_Walker_Helper('cpu')
#    transfer_bipedal_to_multiwalker(model.agents[0])
#    print(model.agents[0].state_dict().keys())