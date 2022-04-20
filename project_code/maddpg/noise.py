import numpy as np
import random
import copy

class Noise:

    def __init__(self, size, seed, mu=0., theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.state = None
        self.reset()

    def reset(self):
        self.state = copy.copy(self.mu)

    def sample(self):
        x = self.state
        dW = np.random.randn(*x.shape)
        dx = self.theta * (self.mu - x) + self.sigma * dW
        self.state = x + dx
        return self.state
