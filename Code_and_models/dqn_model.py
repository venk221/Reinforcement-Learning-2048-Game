import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    
    
    def __init__(self, size_state=4, num_actions=4):

        # Small network for testing
        super(DQN, self).__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(size_state*size_state, 256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.Sigmoid(),
            nn.Linear(256, num_actions)
        )

    def forward(self, x):
        x = self.network(x)
        return x
