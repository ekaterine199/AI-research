import torch
import torch.nn as nn

class PendulumNet(nn.Module):
    def __init__(self, input_dim=1, hidden_layers=3, hidden_units=32):
        super(PendulumNet, self).__init__()
        
        layers = []
        # Input layer
        layers.append(nn.Linear(input_dim, hidden_units))
        layers.append(nn.Tanh())
        
        # Hidden layers
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.Tanh())
            
        # Output layer
        layers.append(nn.Linear(hidden_units, 1))
        
        self.net = nn.Sequential(*layers)

    def forward(self, t):
        return self.net(t)