import torch.nn as nn
import torch
import os

class Combined_Walker_Helper:
    def __init__(self, device, k_levels=1, n_agents=2, use_mlp=False):
        self.num_agents = n_agents
        self.device = device
        self.agents = [Walker_PolicyHelper(self.get_neighbors(agent_ix), use_mlp) for agent_ix in range(self.num_agents)]
        #for agent in self.agents:
        #    transfer_bipedal_to_multiwalker(agent, True)
        self.k_levels = k_levels

    def get_neighbors(self, agent_ix):
        if agent_ix == 0 or agent_ix == self.num_agents - 1:
            return 1
        else:
            return 2

    def eval_experiment(self, experiment_name):
        with open(os.path.join('..','..','experiments','final_models', 'walker', experiment_name, 'combined_model.pt'), 'rb') as f:
            d = torch.load(f, map_location=self.device)
        self.load_state_dicts(d['policy'])

    def parameters(self):
        return [model.parameters() for model in self.agents]

    def to(self, device):
        for agent in self.agents:
            agent.to(device)

    def state_dicts(self):
        return [model.state_dict() for model in self.agents]

    def load_state_dicts(self, state_dicts):
        for ix, model in enumerate(self.agents):
            model.load_state_dict(state_dicts[ix])

    def eval(self):
        for model in self.agents:
            model.eval()

    def __call__(self, observations, adj_matrix):
        """

        :param adj_matrix: torch.Tensor of shape: batch_seg x agents x agents
        :param observations: list of length agents, where element is: batch_seg x obs_shape
        :return:
        """
        k_levels = self.k_levels
        num_batches = observations[0].shape[0]

        policy_initial = []
        state_vals = []
        for agent_ix in range(self.num_agents):
            initial_dist, state_val = self.agents[agent_ix].forward(observations[agent_ix], 0, None)
            state_vals.append(state_val)
            policy_initial.append(initial_dist)

        for k in range(0, k_levels):
            output_dist = []
            # output_dist = torch.zeros(size=policy_initial.shape)
            for agent_ix in range(self.num_agents):
                if agent_ix == 0:
                    batched_neighbors = policy_initial[1].unsqueeze(1)
                elif agent_ix == self.num_agents - 1:
                    batched_neighbors = policy_initial[self.num_agents - 2].unsqueeze(1)
                else:
                    batched_neighbors = torch.cat([policy_initial[agent_ix-1].unsqueeze(1), policy_initial[agent_ix+1].unsqueeze(1)], dim=1)
                current_agent_dist = policy_initial[agent_ix] # batch_seg x latent_shape
                latent_vec = self.agents[agent_ix].forward(current_agent_dist, 1, batched_neighbors)
                output_dist.append(latent_vec) # batch_seq x latent_shape
                #output_dist[:, agent_ix] = latent_vec
            policy_initial = output_dist

        final_actions = []
        for agent_ix in range(0, self.num_agents):
            final_agent_latent = policy_initial[agent_ix]
            final_action_dist = self.agents[agent_ix].forward(final_agent_latent, 2, None) # batch_seg x 4 (action_space)
            final_actions.append(final_action_dist) # batch_seg x 4 (action_space)
        return final_actions, state_vals

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Walker_PolicyHelper(nn.Module):
    def __init__(self, num_neighbors, use_mlp):
        super().__init__()

        encoding_size = 256
        policy_latent_size = 30
        action_space_n = 4

        self.down_sampler = nn.Sequential(
            nn.Linear(31, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, encoding_size),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(encoding_size, policy_latent_size),
            nn.ReLU(),
        )

        self.v_net = nn.Sequential(
            nn.Linear(encoding_size, 1),
            nn.Tanh(),
        )

        self.to_means = nn.Sequential(
            nn.Linear(policy_latent_size, action_space_n),
            nn.Tanh(),
        )

        if use_mlp:
            print('making a linear recurrent policy')
            self.recurr_policy = nn.Sequential(
                nn.Linear(policy_latent_size*(num_neighbors+1), policy_latent_size),
                nn.ReLU(),
            )
            self.apply(init_params)
            self.init_recurr_policy()
        else:
            print('making a gru recurrent policy')
            self.apply(init_params)
            self.recurr_policy = nn.GRU(input_size=policy_latent_size, hidden_size=policy_latent_size, batch_first=True)

    def init_recurr_policy(self):
        self.recurr_policy[0].weight.data.copy_(torch.cat([torch.eye(30), torch.zeros(30, 30)], dim=1))

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
        batch_size, current_dim = observation.shape
        input_tensor = torch.zeros((batch_size, 31)).to(observation.device)
        input_tensor[:, :current_dim] = observation

        encoded = self.down_sampler(input_tensor)
        policy_distribution = self.policy(encoded)
        state_vals = self.v_net(encoded)

        return (policy_distribution, state_vals)

    def forward_communicate(self, policy_dist, neighbors):
        """
        Modify latent vector distribution using neighboring distributions
        :param policy_dist: batchxlatent_size
        :param neighbors: batchxnum_neighborsxlatent_size
        :return: batchxlatent_size
        """
        #print(type(self.recurr_policy), isinstance(self.recurr_policy, nn.Linear))
        if isinstance(self.recurr_policy, nn.Sequential):
            batch, num_neighbors, latent_size = neighbors.shape
            flatten_neighbors = neighbors.reshape((batch, num_neighbors*latent_size))
            return self.recurr_policy(torch.cat([policy_dist, flatten_neighbors], dim=1))
        else:
            _, hn = self.recurr_policy(neighbors, policy_dist.unsqueeze(0))
            return hn.squeeze(0)

    def forward_probs(self, latent_vector):
        means = self.to_means(latent_vector)
        return means

if __name__ == '__main__':
    a = torch.rand((12, 30))
    b = torch.rand((12, 1, 30))
    mod = Walker_PolicyHelper(1, use_mlp=True)
    torch.testing.assert_allclose(mod.forward_communicate(a, b), a)
    print('all good!')