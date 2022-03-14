
import torch.nn as nn
import torch

class Combined_StarCraft_Helper:
    def __init__(self, device, env_name):
        self.device=device
        if env_name == '3m':
            model_cls = Marines
            self.num_agents = 3
        else:
            raise Exception(f"{env_name} isnt supported")
        self.agents = [model_cls().to(device) for _ in range(self.num_agents)]

    def parameters(self):
        return [model.parameters() for model in self.agents]

    def state_dicts(self):
        return [model.state_dict() for model in self.agents]

    def load_state_dicts(self, state_dicts, map_location='cpu'):
        return [model.load_state_dict(state_dicts[ix]) for ix, model in enumerate(self.agents)]

    def eval(self):
        for model in self.agents:
            model.eval()

    def train(self):
        for model in self.agents:
            model.train()

    def __call__(self, observations, adj_matrix, available_actions):
        return self.get_actions(observations, available_actions)

    def get_actions(self, observations, available_actions):
        output_actions = []
        output_values = []
        for agent_ix in range(0, self.num_agents):
            action_dist, v = self.agents[agent_ix].get_actions_state_vals(observations[agent_ix], available_actions[agent_ix])
            output_actions.append(action_dist)
            output_values.append(v)
        return output_actions, output_values

    def get_neighbor_predictions(self, observations, prev_neighbor_actions):
        output_distribution_neighbors = []
        for agent_ix in range(0, self.num_agents):
            neighbor_dists = self.agents[agent_ix].get_actions_others(observations[agent_ix], prev_neighbor_actions[agent_ix])
            output_distribution_neighbors.append(neighbor_dists)
        return output_distribution_neighbors

class Marines(nn.Module):
    def __init__(self):
        super().__init__()
        encoding_size = 200
        policy_latent_size = 50
        action_space_n = 6 + 3

        self.down_sampler = nn.Sequential(
            nn.Linear(30, 300),
            nn.ReLU(),
            nn.Linear(300, encoding_size),
            nn.ReLU(),
        )

        self.fc_e = nn.Sequential(
            nn.Linear(encoding_size, 150),
            nn.ReLU(),
        )

        self.pi_e = nn.Sequential(
            nn.Linear(150, policy_latent_size),
            nn.ReLU(),
            nn.Linear(policy_latent_size, action_space_n),
            nn.ReLU(),
        )
        self.softmax = nn.Softmax(dim=-1)

        self.v_e = nn.Sequential(
            nn.Linear(150, 1),
            nn.Tanh(),
        )

        self.fc_moa = nn.Sequential(
            nn.Linear(encoding_size, policy_latent_size),
            nn.ReLU(),
            nn.Linear(policy_latent_size, action_space_n),
            nn.ReLU(),
        )

        self.gru = nn.GRU(action_space_n, action_space_n, batch_first=True)

    def get_actions_state_vals(self, input_state, available_actions):
        down_samp = self.down_sampler(input_state)

        latent_e = self.fc_e(down_samp)
        output_logits = self.pi_e(latent_e)
        output_logits[~available_actions] = -2e10
        output_dist = self.softmax(output_logits)
        v = self.v_e(latent_e)
        return output_dist, v

    def get_actions_others(self, input_state, prev_neighbor_action_dists):
        down_samp = self.down_sampler(input_state)

        latent_moa = self.fc_moa(down_samp)
        (output_logits_others, _) = self.gru(prev_neighbor_action_dists, latent_moa.unsqueeze(0))
        return output_logits_others