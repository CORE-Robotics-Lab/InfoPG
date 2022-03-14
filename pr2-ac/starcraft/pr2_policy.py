import torch
import torch.nn as nn

class Combined_Starcraft_Helper:
    def __init__(self, device):
        self.device=device
        self.num_agents = 3
        self.agents = [Marines().to(device) for _ in range(self.num_agents)]

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

    def __call__(self, observations, available_actions, adj_matrix,):
        """

        :param adj_matrix: torch.Tensor of shape: batch_seg x agents x agents
        :param observations: list of length agents, where element is: batch_seg x obs_shape
        :return:
        """
        if available_actions is not None:
            return [self.get_action(observations[i], available_actions[i], i) for i in range(0, self.num_agents)]
        else:
            return [self.get_action(observations[i], None, i) for i in range(0, self.num_agents)]

    def get_action(self, obs, available_actions, agent_ix):
        return self.agents[agent_ix].compute_action(obs, available_actions)

    def get_moa(self, next_obs, action_taken, num_particle_samples, agent_ix):
        return self.agents[agent_ix].compute_moa(next_obs, action_taken, num_particle_samples)

    def get_jointq(self, obs, my_action_dist, other_action_dist, agent_ix):
        return self.agents[agent_ix].compute_jointq(obs, my_action_dist, other_action_dist)


class Marines(nn.Module):
    def __init__(self):
        encoding_size = 300
        policy_latent_size = 50
        obs_shape_n = 30
        self.moa_randomness_size = 10
        self.num_neighbors = 2
        self.action_space_n = 6+3

        super().__init__()
        self.down_sampler = nn.Sequential(
            nn.Linear(30, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, 150),
            nn.ReLU(),
        )

        self.fc_e = nn.Sequential(
            nn.Linear(150, policy_latent_size),
            nn.ReLU(),
        )

        self.pi_e = nn.Sequential(
            nn.Linear(policy_latent_size, self.action_space_n),
        )

        self.softmax = nn.Softmax(dim=-1)

        self.rho = nn.Sequential(
            nn.Linear(obs_shape_n + self.action_space_n + self.moa_randomness_size, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, policy_latent_size),
            nn.ReLU(),
            nn.Linear(policy_latent_size, self.action_space_n*self.num_neighbors),
            nn.Unflatten(dim=-1, unflattened_size=(self.num_neighbors, self.action_space_n)),
            nn.ReLU(),
            nn.Softmax(dim=-1),
        )

        self.q_total = nn.Sequential(
            nn.Linear(obs_shape_n + (self.num_neighbors+1)*self.action_space_n, encoding_size),
            nn.Tanh(),
            nn.Linear(encoding_size, policy_latent_size),
            nn.Tanh(),
            nn.Linear(policy_latent_size, 1),
            nn.Tanh(),
        )

        self.qf = nn.Sequential(
            nn.Linear(obs_shape_n + self.action_space_n, policy_latent_size),
            nn.Tanh(),
            nn.Linear(policy_latent_size, 1),
            nn.Tanh()
        )

    def compute_action(self, obs, available_actions):
        down_samp = self.down_sampler(obs)
        latent_e = self.fc_e(down_samp)
        logits = self.pi_e(latent_e)
        if available_actions is not None:
            logits[~available_actions] = -2e10 # make sure that all non-available actions have no chance of being selected
        output_dist = self.softmax(logits)
        return output_dist

    def compute_moa(self, obs, actions, num_particles):
        batch_dim, obs_dim = obs.shape
        shape_latent = (batch_dim, num_particles, self.moa_randomness_size)

        normal_dist = torch.distributions.Normal(loc=0, scale=1)
        latent_randomness = normal_dist.sample(shape_latent).to(obs.device)

        obs = obs.unsqueeze(1).repeat(1, num_particles, 1)
        actions = actions.unsqueeze(1).repeat(1, num_particles, 1)

        opponent_action_dist = self.rho(torch.cat([obs, actions, latent_randomness], dim=-1))
        batch_dim, num_particles, num_neighbors, num_actions = opponent_action_dist.shape

        #return dim: batch_dimxnum_particlesxneighborsxaction
        return opponent_action_dist.reshape((batch_dim, num_particles, num_neighbors*num_actions))

    def compute_jointq(self, obs, my_action_dist, other_action_dist):
        q_val = self.q_total(torch.cat([obs, my_action_dist, other_action_dist], dim=-1))
        return q_val

    def compute_q(self, obs, action):
        return self.qf(torch.cat([obs, action], dim=-1))



