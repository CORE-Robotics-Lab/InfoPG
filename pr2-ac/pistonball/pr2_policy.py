import torch
import torch.nn as nn


class Combined_Piston_Helper:
    def __init__(self, device):
        self.device=device
        self.num_agents = 5
        self.agents = [Policy(4096, 3, 1).to(device) if (ix == 0 or ix == (self.num_agents - 1)) else Policy(4096, 3, 2).to(device)
                       for ix in range(self.num_agents)]

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

    def __call__(self, observations, adj_matrix,):
        """

        :param adj_matrix: torch.Tensor of shape: batch_seg x agents x agents
        :param observations: list of length agents, where element is: batch_seg x obs_shape
        :return:
        """
        return [self.get_action(observations[i], i) for i in range(0, self.num_agents)]

    def get_action(self, obs, agent_ix):
        return self.agents[agent_ix].compute_action(obs)

    def get_moa(self, next_obs, action_taken, num_particle_samples, agent_ix):
        return self.agents[agent_ix].compute_moa(next_obs, action_taken, num_particle_samples)

    def get_jointq(self, obs, my_action_dist, other_action_dist, agent_ix):
        return self.agents[agent_ix].compute_jointq(obs, my_action_dist, other_action_dist)


class Policy(nn.Module):
    def __init__(self, obs_shape_n, action_space_n, num_neighbors):
        encoding_size = 30
        policy_latent_size = 10
        self.moa_randomness_size = 3
        self.num_neighbors = num_neighbors
        self.action_space_n = action_space_n

        super().__init__()
        self.down_sampler = nn.Sequential(
            nn.Linear(obs_shape_n, encoding_size),
            nn.ReLU()
        )

        self.fc_e = nn.Sequential(
            nn.Linear(encoding_size, policy_latent_size),
            nn.ReLU(),
        )

        self.pi_e = nn.Sequential(
            nn.Linear(policy_latent_size, action_space_n),
            nn.ReLU(),
            nn.Softmax(dim=-1),
        )

        self.rho = nn.Sequential(
            nn.Linear(obs_shape_n + action_space_n + self.moa_randomness_size, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, policy_latent_size),
            nn.ReLU(),
            nn.Linear(policy_latent_size, action_space_n*num_neighbors),
            nn.Unflatten(dim=-1, unflattened_size=(num_neighbors, action_space_n)),
            nn.ReLU(),
            nn.Softmax(dim=-1),
        )

        self.q_total = nn.Sequential(
            nn.Linear(obs_shape_n + (num_neighbors+1)*action_space_n, encoding_size),
            nn.Tanh(),
            nn.Linear(encoding_size, policy_latent_size),
            nn.Tanh(),
            nn.Linear(policy_latent_size, 1),
            nn.Tanh(),
        )

        self.qf = nn.Sequential(
            nn.Linear(obs_shape_n + action_space_n, policy_latent_size),
            nn.Tanh(),
            nn.Linear(policy_latent_size, 1),
            nn.Tanh()
        )

        self.num_neighbors = num_neighbors

    def compute_action(self, obs):
        down_samp = self.down_sampler(obs)
        latent_e = self.fc_e(down_samp)
        output_dist = self.pi_e(latent_e)
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


if __name__ == '__main__':
    obs_dim = 6525
    action_dim = 3
    num_neighbors = 1
    policy = Policy(obs_dim, action_dim, num_neighbors)
    obs = torch.rand((1, obs_dim))
    action = policy.compute_action(obs)
    actions_other = policy.compute_moa(obs)
    q = policy.compute_jointq(obs, action, actions_other)
    print(action.shape, actions_other.shape, q.shape)


