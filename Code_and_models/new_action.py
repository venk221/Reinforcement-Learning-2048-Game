import torch
import numpy as np

def make_action(online_net, observation, device):
    CPU = torch.device('cpu')
    obs_t = torch.as_tensor(observation.copy(), dtype=torch.float32)
    q_values = (obs_t.unsqueeze(0))
    online_net.to(device)
    action_t = online_net(q_values.to(device))
    action = np.argmax(action_t.to(CPU).detach().numpy())
    return action 