from policy_base import BasePolicy, BasePolicy_MOA
import torch.nn as nn
import torch

class PistonPolicyCASE(BasePolicy):
    def __init__(self, device, model_state_dict=None, constant=False):
        super().__init__(device, 'normal')
        self.policy = Piston_PolicyHelperCASE()
        if constant:
            for name, param in self.policy.named_parameters():
                nn.init.constant_(param, 0.5)
        if model_state_dict is not None:
            self.policy.load_state_dict(model_state_dict)
        self.policy.to(device)

class Piston_PolicyHelperCASE(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sampler = nn.Sequential(
            nn.Linear(4096, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(50, 25),
            nn.Linear(25, 10),
            nn.ReLU(),
        )
        self.v_net = nn.Sequential(
            nn.Linear(50, 1),
            nn.Tanh(), #using tanh because we can have negative rewards
        )
        self.to_probs = nn.Sequential(
            nn.Linear(10, 3),
            nn.ReLU(),
            nn.Softmax(dim=-1),
        )
        self.recurr_policy = nn.GRUCell(input_size=10, hidden_size=10)

    def forward(self, observation, step, neighbors=None):
        if step == 0:
            return self.forward_initial(observation)
        elif step == 1:
            return self.forward_communicate(observation, neighbors)
        elif step == 2:
            return self.forward_probs(observation)
        else:
            raise Exception('Incorrect step number for forward prop, should be: 0,1,2')

    def forward_initial(self, observation):
        encoded = self.down_sampler(observation)

        policy_distribution = self.policy(encoded)
        state_vals = self.v_net(encoded)

        return (policy_distribution, state_vals)

    def forward_communicate(self, policy_dist, neighbors):
        """
        Modify latent vector distribution using neighboring distributions
        :param policy_dist: batchxlatent_size
        :param neighbors: batchxnum_neighborsx[latent_size, str]
        :return: batchxlatent_size
        """
        num_batches = len(neighbors)
        assert len(policy_dist) == num_batches, 'Error here'
        batch_outputs = []
        batch_temp_outputs = [[] for _ in range(num_batches)]
        for batch_ix in range(0, num_batches):
            neighbor_dists = neighbors[batch_ix] #<- all neighbors at some batch
            num_neighbors = len(neighbor_dists)
            hx = policy_dist[batch_ix].unsqueeze(0) # <- initial hidden state
            batch_temp_outputs[batch_ix].append(hx)
            for neighbor_ix in range(0, num_neighbors):
                neighbor_dist, name, cop_batch_ix = neighbor_dists[neighbor_ix]
                batch_neighbor_dist = neighbor_dist.unsqueeze(0)
                assert hx.shape == batch_neighbor_dist.shape, '%s and %s' % (hx.shape, batch_neighbor_dist.shape)
                assert cop_batch_ix == batch_ix, 'Error here'
                hx = self.recurr_policy(batch_neighbor_dist, hx)
                batch_temp_outputs[batch_ix].append(hx)
            batch_outputs.append(hx)
        final_out = torch.cat(batch_outputs, dim=0)
        final_temp_out = torch.cat([torch.cat(batch_temp_outputs[ix], dim=0).unsqueeze(0) for ix in range(num_batches)], dim=0) # batchxseqlenxhiddensize
        return final_out, final_temp_out

    def forward_probs(self, latent_vector):
        probs = self.to_probs(latent_vector)
        return probs

class MOAPolicyCASE(BasePolicy_MOA):
    def __init__(self, device, model_state_dict=None):
        super().__init__(device)
        self.policy = MOA_PolicyCASEHelper()
        if model_state_dict is not None:
            self.policy.load_state_dict(model_state_dict)
        self.policy.to(device)

class MOA_PolicyCASEHelper(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_sampler = nn.Sequential(
            nn.Linear(4096, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
        )

        self.fc_e = nn.Sequential(
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.ReLU(),
        )

        self.pi_e = nn.Sequential(
            nn.Linear(10, 3),
            nn.ReLU(),
            nn.Softmax(dim=-1),
        )

        self.v_e = nn.Sequential(
            nn.Linear(10, 1),
            nn.Tanh(),
        )

        self.fc_moa = nn.Sequential(
            nn.Linear(50, 15),
            nn.ReLU(),
            nn.Linear(15, 3),
            nn.ReLU(),
        )

        self.gru = nn.GRU(3,3, batch_first=True)

    def forward(self, input_state, step, prev_neighbor_action_dists):
        down_samp = self.down_sampler(input_state)
        latent_e = self.fc_e(down_samp)
        output_dist = self.pi_e(latent_e)
        v = self.v_e(latent_e)

        latent_moa = self.fc_moa(down_samp)
        (output_actions_other_agents, _) = self.gru(prev_neighbor_action_dists, latent_moa.unsqueeze(0))
        return output_dist, v, output_actions_other_agents

