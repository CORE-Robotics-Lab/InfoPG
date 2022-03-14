from wrapped_starcraft_env import StarCraft2Env_MAF
import torch
from distributed_starcraft_policy import Combined_StarCraft_Helper
import os
import ray
from distributed_starcraft_env import StarCraft_Worker

def send_weights(model, worker_ref):
    state_dicts = model.state_dicts()
    new_state_dicts = [{} for _ in range(len(state_dicts))]
    for agent_ix in range(0, len(state_dicts)):
        for param_name in state_dicts[agent_ix].keys():
            if isinstance(state_dicts[agent_ix][param_name], torch.Tensor):
                new_state_dicts[agent_ix][param_name] = state_dicts[agent_ix][param_name].clone().to('cpu')
    weight_id = ray.put(new_state_dicts)
    ray.get(worker_ref.set_model_weights.remote(weight_id))

def visualize(experiment_dir):
    device='cpu'
    model = Combined_StarCraft_Helper('cpu')
    with open(os.path.join('experiments', 'starcraft', experiment_dir, 'best', 'combined_model.pt'), 'rb') as f:
        d = torch.load(f, map_location=device)
        model.load_state_dicts(d['policy'])

    ray.init(num_cpus = 1, _node_ip_address="0.0.0.0")
    env_params = {
        'map_name': '3m', 'reward_only_positive': False
    }
    worker_ref = StarCraft_Worker.remote(1, env_params, time_penalty=0.0, device='cpu')
    send_weights(model, worker_ref)
    for i in range(0, 1000):
        ray.get(worker_ref.train.remote(sleep_time=0.5))
    ray.get(worker_ref.close.remote())
    ray.shutdown()

if __name__ == '__main__':
    visualize('2021-07-06 18_52_45')