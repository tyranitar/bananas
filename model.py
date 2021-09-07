from contextlib import contextmanager
import torch.nn.functional as F
import torch.nn as nn
import torch
import math

# Initializes a layer with normally-distributed weights.
def normal_weights(layer):
    classname = layer.__class__.__name__

    if classname.find('Linear') != -1:
        n = layer.in_features
        y = 1.0 / math.sqrt(n)
        layer.weight.data.normal_(0, y)

# A Dueling DQN.
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed=1337):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        state_val_net_layer_dims = [
            state_size,
            128,
            32,
            # 1
        ]

        advantage_net_layer_dims = [
            state_size,
            128,
            32,
            # 4
        ]

        # V(s)
        self.state_val_net = nn.Sequential(
            *self.gen_linear_layers(state_val_net_layer_dims),
            nn.Linear(state_val_net_layer_dims[-1], 1)
        )

        # A(s, a)
        self.advantage_net = nn.Sequential(
            *self.gen_linear_layers(advantage_net_layer_dims),
            nn.Linear(advantage_net_layer_dims[-1], action_size)
        )

        self.apply(normal_weights)

    def gen_linear_layers(self, layer_dims):
        return [
            nn.Sequential(
                nn.Linear(layer_dims[i], layer_dims[i + 1]),
                nn.BatchNorm1d(layer_dims[i + 1]),
                nn.ReLU(),
            )
            for i in range(len(layer_dims) - 1)
        ]

    def forward(self, state):
        state_vals = self.state_val_net(state)
        advantages = self.advantage_net(state)

        # Q(s, a) = V(s) + A(s, a) - mean(A(s, a'))
        return state_vals + advantages - advantages.mean()

    # Use this to interact with the environment
    # since action ranks don't change with V(s).
    def get_advantages(self, state):
        return self.advantage_net(state)

    @contextmanager
    def eval_no_grad(self):
        with torch.no_grad():
            try:
                self.eval()
                yield
            finally:
                self.train()
