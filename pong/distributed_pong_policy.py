import torch.nn as nn
import torch

class Combined_Pong_Helper:
    def __init__(self, device, k_levels=1):
        self.num_agents = 2
        self.agents = [Pong_PolicyHelper().to(device) for _ in range(self.num_agents)]
        self.k_levels = k_levels

    def parameters(self):
        return [model.parameters() for model in self.agents]

    def state_dicts(self):
        return [model.state_dict() for model in self.agents]

    def load_state_dicts(self, state_dicts):
        return [model.load_state_dict(state_dicts[ix]) for ix, model in enumerate(self.agents)]

    def consensus_update(self):
        paddle_0 = self.agents[0].v_net.state_dict()
        paddle_1 = self.agents[1].v_net.state_dict()

        avg_state_dict_0 = {}
        avg_state_dict_1 = {}
        for k in paddle_0.keys():
            avg_state_dict_0[k] = (paddle_0[k] + paddle_1[k])/2
            avg_state_dict_1[k] = (paddle_0[k] + paddle_1[k])/2

        self.agents[0].v_net.load_state_dict(avg_state_dict_0)
        self.agents[1].v_net.load_state_dict(avg_state_dict_1)

    def __call__(self, observations, prev_latent_vectors):
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
                    batched_neighbors = prev_latent_vectors[:, 1]
                else:
                    batched_neighbors = prev_latent_vectors[:, 0]
                current_agent_dist = policy_initial[agent_ix] # batch_seg x latent_shape
                latent_vec = self.agents[agent_ix].forward(current_agent_dist, 1, batched_neighbors.unsqueeze(1))
                output_dist.append(latent_vec) # batch_seq x 1 x latent_shape
                #output_dist[:, agent_ix] = latent_vec
            policy_initial = output_dist

        final_actions = []
        for agent_ix in range(0, self.num_agents):
            final_agent_latent = policy_initial[agent_ix]
            final_action_dist = self.agents[agent_ix].forward(final_agent_latent, 2, None) # batch_seg x 3 (action_space)
            final_actions.append(final_action_dist) # batch_seg x 3 (action_space)
        return final_actions, state_vals, policy_initial

# Function from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/model.py
def init_params(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Pong_PolicyHelper(nn.Module):
    def __init__(self):
        encoding_size = 300
        policy_latent_size = 30
        action_space_n = 3

        super().__init__()
        self.down_sampler = nn.Sequential(
            nn.Linear(6525, encoding_size),
            nn.ReLU()
        )
        self.policy = nn.Sequential(
            nn.Linear(encoding_size, policy_latent_size),
            nn.ReLU(),
        )
        self.v_net = nn.Sequential(
            nn.Linear(encoding_size, 1),
            nn.Tanh(), #using tanh because we can have negative rewards
        )
        self.to_probs = nn.Sequential(
            nn.Linear(policy_latent_size, action_space_n),
            nn.ReLU(),
            nn.Softmax(dim=-1),
        )

        self.apply(init_params)
        self.recurr_policy = nn.Sequential(
            nn.Linear(policy_latent_size*2, policy_latent_size, bias=False),
            nn.ReLU(),
        )
        self.init_recurr_policy()

    def init_recurr_policy(self):
        out_dim, in_dim = self.recurr_policy[0].weight.shape
        self.recurr_policy[0].weight.data = torch.cat([torch.eye(out_dim), torch.zeros((out_dim, out_dim))], dim=1)
        self.recurr_policy[0].weight.data.requires_grad = False

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
        :param neighbors: batchxnum_neighborsxlatent_size
        :return: batchxlatent_size
        """
        #_, hn = self.recurr_policy(neighbors, policy_dist.unsqueeze(0))
        #  return hn.squeeze(0)
        merged = torch.cat([policy_dist, neighbors.squeeze(1)], dim=1)
        hn = self.recurr_policy(merged)
        return hn

    def forward_probs(self, latent_vector):
        probs = self.to_probs(latent_vector)
        return probs

if __name__ == '__main__':
    pong_helper = Pong_PolicyHelper()

    x = torch.rand((16, 30))
    y = torch.rand((16, 1, 30))


    out = pong_helper.forward_communicate(x, y)
    torch.testing.assert_allclose(x, out)
    print('all good!')