import numpy as np
import random
from collections import deque, namedtuple
import torch

class ReplayBuffer:
    """Replaybuffer to store experiences."""

    def __init__(self, buffer_size, batch_size,  device, random_seed = 0):

        """Initialize a MemoryBuffer object.
        :param buffer_size: number of samples
        :param batch_size: batch_size when sampling random entries
        :param seed: random seed

        """
        random.seed(random_seed)
        self.device = device
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)
        experiences = list(map(lambda x: np.asarray(x), zip(*experiences)))
        states, actions, rewards, next_states, dones = [torch.from_numpy(e).float().to(self.device) for e in experiences]
        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)