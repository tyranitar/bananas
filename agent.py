import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
import torch

from model import QNetwork

###########################
# Begin hyper-parameters. #
###########################

BUFFER_SIZE = 100000
BATCH_SIZE = 64
# Learning rate.
ALPHA = 0.0005
# Discount factor.
GAMMA = 0.99
# Soft target update.
TAU = 0.001
# Target update period.
UPDATE_EVERY = 4
# Double DQN.
USE_DDQN = True

# Minimum priority.
E = 0.001
# Priority exponent.
A = 0.5
# Importance sampling weight exponent.
B = 1

# Don't reduce since we need to scale
# each loss by a sampling weight.
loss_fn = nn.SmoothL1Loss(reduce=False)

#########################
# End hyper-parameters. #
#########################

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    def __init__(self, state_size, action_size, seed=1337):
        random.seed(seed)

        self.state_size = state_size
        self.action_size = action_size

        self.local_net = QNetwork(state_size, action_size, seed).to(device)
        self.target_net = QNetwork(state_size, action_size, seed).to(device)

        # Sync the local and target networks.
        self.soft_update_target(1)

        self.optimizer = optim.Adam(self.local_net.parameters(), lr=ALPHA)
        self.memory = ReplayBuffer(state_size, action_size, BUFFER_SIZE, BATCH_SIZE, seed)

        self.time_step = 0

    def act(self, state, epsilon=0):
        if random.random() < epsilon:
            return random.choice(range(self.action_size))

        # Else, choose the greedy action.

        state = torch.from_numpy(state).float().unsqueeze(0).to(device)

        with self.local_net.eval_no_grad():
            action_vals = self.local_net.get_advantages(state)

        return action_vals.cpu().argmax().item()

    def step(self, state, action, reward, next_state, done):
        # Since the max priority will be 0 before any learning happens, let's
        # default to a value of 1, which is where the reward is clipped.
        priority = self.memory.get_max_priority() if self.memory.can_sample() else 1

        self.memory.push(
            torch.from_numpy(state),
            action,
            reward,
            torch.from_numpy(next_state),
            int(done),
            priority,
        )

        self.time_step = (self.time_step + 1) % UPDATE_EVERY

        if self.time_step == 0 and self.memory.can_sample():
            experiences = self.memory.sample()
            self.learn(experiences)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones, indices, sampling_probs = experiences

        if USE_DDQN:
            local_net_choices = self.local_net(next_states).argmax(dim=1).unsqueeze(1)
            target_action_vals = self.target_net(next_states).gather(1, local_net_choices).view(-1)
        else:
            target_action_vals = self.target_net(next_states).max(dim=1)[0]

        # Compute TD targets.
        target_action_vals = target_action_vals.detach()
        td_targets = dones * rewards + (1 - dones) * (rewards + GAMMA * target_action_vals)

        # Compute current values.
        action_vals = self.local_net(states)
        td_outputs = action_vals.gather(1, actions.unsqueeze(1)).view(-1)

        # Compute sampling weights.
        N = self.memory.get_len_sample_space()
        sampling_weights = ((1 / N) * (1 / sampling_probs)).pow(B)
        sampling_weights /= sampling_weights.max()

        # Compute and scale losses.
        loss = loss_fn(td_outputs, td_targets)
        loss = sampling_weights * loss
        loss = loss.mean()

        # Take a gradient descent step.
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        # Soft-update the target network.
        self.soft_update_target(TAU)

        # Update the priorities of the sampled experience tuples.
        priorities = ((td_targets - td_outputs.detach()).abs() + E).pow(A)
        self.memory.update_priorities(indices, priorities)

    def soft_update_target(self, tau):
        for target_params, local_params in zip(
            self.target_net.parameters(),
            self.local_net.parameters(),
        ):
            target_params.data.copy_(
                tau * local_params.data + \
                (1 - tau) * target_params.data
            )

class ReplayBuffer():
    def __init__(self, state_size, action_size, buffer_size, batch_size, seed=1337):
        assert batch_size < buffer_size

        random.seed(seed)

        self.batch_size = batch_size

        self.states = RollingTensor((buffer_size, state_size))
        self.actions = RollingTensor((buffer_size,), torch.long)
        self.rewards = RollingTensor((buffer_size,))
        self.next_states = RollingTensor((buffer_size, state_size))
        self.dones = RollingTensor((buffer_size,))

        self.priorities = RollingTensor((buffer_size,))

    def push(self, state, action, reward, next_state, done, priority):
        self.states.push(state)
        self.actions.push(action)
        self.rewards.push(reward)
        self.next_states.push(next_state)
        self.dones.push(done)

        self.priorities.push(priority)

    def get_len_sample_space(self):
        # Since all tensors must have the same number
        # of items, any one of them will do.
        return self.states.get_num_items()

    def can_sample(self):
        return self.get_len_sample_space() >= self.batch_size

    def get_priority_sample_space(self):
        return self.priorities[:self.get_len_sample_space()]

    def get_max_priority(self):
        return self.get_priority_sample_space().max()

    def sample(self):
        assert self.can_sample()

        indices = self.get_priority_sample_space().multinomial(self.batch_size)

        states = self.states[indices].to(device)
        actions = self.actions[indices].to(device)
        rewards = self.rewards[indices].to(device)
        next_states = self.next_states[indices].to(device)
        dones = self.dones[indices].to(device)

        sampling_probs = self.priorities[indices] / self.get_priority_sample_space().sum()

        return (states, actions, rewards, next_states, dones, indices, sampling_probs)

    def update_priorities(self, indices, new_priorities):
        self.priorities[indices] = new_priorities

# Rolling tensor that's memory-efficient and performant.
class RollingTensor():
    def __init__(self, dims, dtype=torch.float32):
        self.tensor = torch.zeros(dims, dtype=dtype)
        self.len = dims[0]
        self.num_items = 0
        self.i = 0

    def push(self, item):
        self.tensor[self.i] = item

        if self.num_items < self.len:
            self.num_items = self.i + 1

        self.i = (self.i + 1) % self.len

    def get_num_items(self):
        return self.num_items

    def __getitem__(self, key):
        return self.tensor[key]

    def __setitem__(self, key, val):
        self.tensor[key] = val

    def __getslice__(self, start, end):
        return self.tensor[start:end]
