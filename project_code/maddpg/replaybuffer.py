import numpy as np
import random
from collections import namedtuple, deque

epsilon = 0.001

class ReplayBuffer:

    def __init__(self, num_agent, observation_size, action_size, buffer_size, batch_size, device):

        self.buffer = deque(maxlen=buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["observations", "actions", "rewards", "next_observations", "dones"])

        """
        self.num_agent = num_agent
        self.action_size = action_size
        self.observation_size = observation_size
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.data_observation = np.zeros(shape=(self.buffer_size, num_agent, observation_size))
        self.data_actions = np.zeros(shape=(self.buffer_size, num_agent, action_size))
        self.data_rewards = np.zeros(shape=(self.buffer_size, num_agent,))
        self.data_next_observations = np.zeros(shape=(self.buffer_size, num_agent, observation_size))
        self.data_dones = np.zeros(shape=(self.buffer_size, num_agent,), dtype=np.uint8)

        self.next_idx = 0
        self.size = 0
        """
        self.device = device

    def __len__(self):
        #return self.size
        return len(self.buffer)


    def add(self, observation, actions, rewards, next_observation, dones):

        """
        idx = self.next_idx
        self.data_observation[idx] = observation
        self.data_actions[idx] = actions
        self.data_rewards[idx] = rewards
        self.data_next_observations[idx] = next_observation
        self.data_dones[idx] = dones

        self.next_idx = (idx + 1) % self.buffer_size
        self.size = min(self.buffer_size, self.size + 1)
        """

        experience = self.experience(observation, actions, rewards, next_observation, dones)
        self.buffer.append(experience)

    def sample(self):

        """
        indexes = np.random.choice(self.buffer_size, size=self.batch_size, replace=False)
        observations = self.data_observation[indexes,:,:]
        actions = self.data_actions[indexes,:,:]
        rewards = self.data_rewards[indexes,:]
        next_observations = self.data_next_observations[indexes,:,:]
        dones = self.data_dones[indexes,:]
        """


        experiences = random.sample(self.buffer, k=self.batch_size)
        observations, actions, rewards, next_observations, dones = list(map(lambda x: np.asarray(x), zip(*experiences)))
        return observations, actions, rewards, next_observations, dones


class PrioritizedReplayBuffer:

    def __init__(self, num_agent, observation_size, action_size, buffer_size, batch_size, device, alpha, beta, beta_increment_per_sampling):

        self.num_agent = num_agent
        self.action_size = action_size
        self.observation_size = observation_size
        self.batch_size = batch_size
        self.device = device
        self.buffer_size = buffer_size
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        self.priority_sum = [0 for i in range(2 * self.buffer_size)]
        self.priority_min = [float('inf') for i in range(2 * self.buffer_size)]
        self.max_priority = 1


        self.data_observation = np.zeros(shape=(self.buffer_size, num_agent, observation_size))
        self.data_actions = np.zeros(shape=(self.buffer_size, num_agent, action_size))
        self.data_rewards = np.zeros(shape=(self.buffer_size, num_agent,))
        self.data_next_observation = np.zeros(shape=(self.buffer_size, num_agent, observation_size))
        self.data_dones = np.zeros(shape=(self.buffer_size, num_agent,), dtype=np.bool)

        self.next_idx = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, observation, actions, rewards, next_observation, dones):

        idx = self.next_idx
        self.data_observation[idx] = observation
        self.data_actions[idx] = actions
        self.data_rewards[idx] = rewards
        self.data_next_observation[idx] = next_observation
        self.data_dones[idx] = dones

        self.next_idx = (idx + 1) % self.buffer_size

        self.size = min(self.buffer_size, self.size + 1)

        priority_alpha = self.max_priority ** self.alpha

        self.set_priority_min(idx, priority_alpha)
        self.set_priority_sum(idx, priority_alpha)

    def set_priority_min(self, idx, priority_alpha):
        idx += self.buffer_size
        self.priority_min[idx] = priority_alpha

        while idx >= 2:
            idx = idx // 2
            self.priority_min[idx] = min(self.priority_min[2 * idx], self.priority_min[2 * idx + 1])

    def set_priority_sum(self, idx, priority):
        idx += self.buffer_size
        self.priority_sum[idx] = priority
        # print(priority)

        while idx >= 2:
            idx = idx // 2
            self.priority_sum[idx] = self.priority_sum[2 * idx] + self.priority_sum[2 * idx + 1]

    def min(self):
        return self.priority_min[1]

    def sum(self):
        return self.priority_sum[1]

    def is_full(self):
        return self.buffer_size == self.size

    def sample(self):

        indexes = np.zeros(shape=self.batch_size, dtype=np.int32)
        ISWeights = np.zeros(shape=self.batch_size, dtype=np.float32)
        observations = np.zeros(shape=(self.batch_size, self.num_agent, self.observation_size), dtype=np.float32)
        actions = np.zeros(shape=(self.batch_size, self.num_agent, self.action_size), dtype=np.float32)
        rewards = np.zeros(shape=(self.batch_size, self.num_agent,), dtype=np.float32)
        next_observations = np.zeros(shape=(self.batch_size, self.num_agent, self.observation_size), dtype=np.float32)
        dones = np.zeros(shape=(self.batch_size, self.num_agent,), dtype=np.float32)


        for i in range(self.batch_size):
            p = random.random() * self.sum()
            idx = self.find_prefix_sum_idx(p)
            indexes[i] = idx
            observations[i] = self.data_observation[idx]
            actions[i] = self.data_actions[idx]
            rewards[i] = self.data_rewards[idx]
            next_observations[i] = self.data_next_observation[idx]
            dones[i] = self.data_dones[idx]


        probability_min = self.min() / self.sum()
        max_weight = (probability_min * self.size) ** (-self.beta)

        # calculate weights
        for i in range(self.batch_size):
            idx = indexes[i]

            probability = self.priority_sum[idx + self.buffer_size] / self.sum()

            weight = (probability * self.size) ** (-self.beta)

            ISWeights[i] = weight / max_weight

        self.beta = min(1, self.beta + self.beta_increment_per_sampling)

        return ISWeights, indexes, observations, actions, rewards, next_observations, dones

    def update_priorities(self, indexes, priorities):
        priorities += epsilon

        for idx, priority in zip(indexes, priorities):

            self.max_priority = max(self.max_priority, priority)

            priority_alpha = priority ** self.alpha

            self.set_priority_min(idx, priority_alpha)
            self.set_priority_sum(idx, priority_alpha)

    def find_prefix_sum_idx(self, prefix_sum):
        idx = 1
        while idx < self.buffer_size:
            if self.priority_sum[idx * 2] > prefix_sum:
                idx = 2 * idx
            else:
                prefix_sum -= self.priority_sum[idx * 2]
                idx = 2 * idx + 1
        return idx - self.buffer_size