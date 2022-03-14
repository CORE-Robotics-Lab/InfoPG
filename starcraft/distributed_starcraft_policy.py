
import torch.nn as nn
import torch

class Combined_StarCraft_Helper:
    def __init__(self, device, env_name, k_levels=1):
        self.device=device
        if env_name == '3m':
            model_cls = Marines
            self.num_agents = 3
        elif env_name == '2c_vs_64zg':
            model_cls = Colossi
            self.num_agents = 2
        else:
            raise Exception(f"{env_name} isnt supported")
        self.agents = [model_cls().to(device) for _ in range(self.num_agents)]
        self.k_levels = k_levels

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

    def consensus_update(self):
        paddle_0 = self.agents[0].v_net.state_dict()
        paddle_1 = self.agents[1].v_net.state_dict()
        paddle_2 = self.agents[2].v_net.state_dict()

        avg_state_dict_0 = {}
        avg_state_dict_1 = {}
        avg_state_dict_2 = {}
        for k in paddle_0.keys():
            avg_state_dict_0[k] = (paddle_0[k] + paddle_1[k] + paddle_2[k])/3
            avg_state_dict_1[k] = (paddle_0[k] + paddle_1[k] + paddle_2[k])/3
            avg_state_dict_2[k] = (paddle_0[k] + paddle_1[k] + paddle_2[k])/3

        self.agents[0].v_net.load_state_dict(avg_state_dict_0)
        self.agents[1].v_net.load_state_dict(avg_state_dict_1)
        self.agents[2].v_net.load_state_dict(avg_state_dict_2)

    def __call__(self, observations, adj_matrix, available_actions):
        """

        :param adj_matrix: torch.Tensor of shape: batch_seg x agents x agents
        :param observations: list of length agents, where element is: batch_seg x obs_shape
        :param observations: list of length agents, where element is: batch_seg x 9 (action_space_n)
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
                current_agent_dist = policy_initial[agent_ix] # batch_seg x latent_shape
                _, latent_size = current_agent_dist.shape
                neighbor_torch_dist = torch.zeros((num_batches, self.num_agents, latent_size))
                for neighbor_ix in range(self.num_agents):
                    mask = adj_matrix[:, agent_ix, neighbor_ix]
                    neighbor_torch_dist[mask, neighbor_ix, :] = policy_initial[neighbor_ix][mask, :]
                neighbor_torch_dist[:, agent_ix, :] = 0
                latent_vec = self.agents[agent_ix].forward(current_agent_dist, 1, neighbor_torch_dist)
                output_dist.append(latent_vec) # batch_seq x latent_shape
            policy_initial = output_dist

        final_actions = []
        for agent_ix in range(0, self.num_agents):
            final_agent_latent = policy_initial[agent_ix]
            final_action_dist = self.agents[agent_ix].forward(final_agent_latent, 2, available_actions[agent_ix]) # batch_seg x 9 (action_space)
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

class Marines(nn.Module):
    def __init__(self):
        encoding_size = 300
        policy_latent_size = 50
        action_space_n = 6 + 3

        super().__init__()
        self.down_sampler = nn.Sequential(
            nn.Linear(30, encoding_size),
            nn.ReLU(),
            nn.Linear(encoding_size, 150),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(150, policy_latent_size),
            nn.ReLU(),
        )
        self.v_net = nn.Sequential(
            nn.Linear(150, 1),
            nn.Tanh(), #using tanh because we can have negative rewards
        )
        self.to_logits = nn.Sequential(
            nn.Linear(policy_latent_size, action_space_n),
        )
        self.softmax = nn.Softmax(dim=-1)

        self.apply(init_params)
        self.recurr_policy = nn.GRU(input_size=policy_latent_size, hidden_size=policy_latent_size, batch_first=True)

    def forward(self, observation, step, neighbors=None):
        if step == 0:
            return self.forward_initial(observation)
        elif step == 1:
            return self.forward_communicate(observation, neighbors)
        elif step == 2:
            return self.forward_probs(observation, neighbors)
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
        _, hn = self.recurr_policy(neighbors, policy_dist.unsqueeze(0))
        return hn.squeeze(0)

    def forward_probs(self, latent_vector, available_actions):
        logits = self.to_logits(latent_vector)
        logits[~available_actions] = -2e10 # make sure that all non-available actions have no chance of being selected
        probs = self.softmax(logits)
        return probs

class Colossi(nn.Module):
    def __init__(self):
        encoding_size = 300
        policy_latent_size = 100
        #action_space_n = 6+64 #6+3
        action_space_n = 6 + 32

        super().__init__()
        self.down_sampler = nn.Sequential(
            nn.Linear(332, 1000),
            nn.ReLU(),
            nn.Linear(1000, 300),
            nn.ReLU(),
        )
        self.policy = nn.Sequential(
            nn.Linear(300, policy_latent_size),
            nn.ReLU(),
        )
        self.v_net = nn.Sequential(
            nn.Linear(300, 1),
            nn.Tanh(), #using tanh because we can have negative rewards
        )
        self.to_logits = nn.Sequential(
            nn.Linear(policy_latent_size, action_space_n),
        )
        self.softmax = nn.Softmax(dim=-1)

        self.apply(init_params)
        """self.recurr_policy = nn.Sequential(
            nn.Linear(2*policy_latent_size, policy_latent_size),
            nn.ReLU(),
        )"""
        self.recurr_policy = nn.GRU(input_size=policy_latent_size, hidden_size=policy_latent_size, batch_first=True)

    def forward(self, observation, step, neighbors=None):
        if step == 0:
            return self.forward_initial(observation)
        elif step == 1:
            return self.forward_communicate(observation, neighbors)
        elif step == 2:
            return self.forward_probs(observation, neighbors)
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
        _, hn = self.recurr_policy(neighbors, policy_dist.unsqueeze(0))
        return hn.squeeze(0)
        #out = self.recurr_policy(torch.cat([policy_dist, neighbors.squeeze(1)], dim=1))
        #return out

    def forward_probs(self, latent_vector, available_actions):
        logits = self.to_logits(latent_vector)
        new_logits = torch.cat([logits[:, 0:6], torch.repeat_interleave(logits[:, 6:], 2, dim=1)], dim=1)
        #print(available_actions)
        new_logits[~available_actions] = -2e10 # make sure that all non-available actions have no chance of being selected
        probs = self.softmax(new_logits)
        #print(probs)
        return probs

if __name__ == '__main__':
    import numpy as np

    model =Combined_StarCraft_Helper(env_name='3m', k_levels=2, device='cpu')
    batch = 10
    observations = [torch.rand((batch, 30)) for _ in range(0, model.num_agents)]
    adj_matrix = np.tile((~np.eye((model.num_agents))[np.newaxis, ...].astype(bool)), (batch, 1, 1))
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.bool)
    available_actions = torch.ones((batch, model.num_agents, 9))
    available_actions = [torch.tensor(available_actions[:, agent_ix], dtype=torch.bool) for agent_ix in range(0, model.num_agents)]
    model(observations, adj_matrix, available_actions)



