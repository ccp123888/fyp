import torch
import numpy as np

from .noise import Noise

from .model import ActorNetwork, CriticNetwork


class Agent:
    def __init__(self, environment, train_config, device):

        self.environment = environment
        self.num_agent = self.environment.num_agent
        self.observation_size = environment.observation_size * environment.observation_history_length
        self.action_size = environment.action_size

        self.mini_batch_size = train_config['mini_batch_size']
        self.actor_learning_rate = train_config['actor_learning_rate']
        self.critic_learning_rate = train_config['critic_learning_rate']
        self.gamma = train_config['discount_gamma']
        self.tau = train_config['soft_update_tau']

        self.device = device

        # Actor networks
        actor_input_size = self.observation_size
        actor_output_size = self.action_size
        self.actor_local = ActorNetwork(actor_input_size, actor_output_size).to(device)
        self.actor_target = ActorNetwork(actor_input_size, actor_output_size).to(device)

        # Critic networks
        critic_input_size = self.num_agent * (self.observation_size + self.action_size)
        critic_output_size = 1  # Only Q_value
        self.critic_local = CriticNetwork(critic_input_size, critic_output_size).to(device)
        self.critic_target = CriticNetwork(critic_input_size, critic_output_size).to(device)

        # Optimizers
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), self.actor_learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), self.critic_learning_rate)

        # Initial Random Process
        self.noise = Noise(self.action_size, 1)

    def reset_noise(self):
        self.noise.reset()

    def flatten(self, unflattened_tensor):

        batch_size, num_agents, vector_size = unflattened_tensor.shape
        flattened_tensor = unflattened_tensor.view(batch_size, num_agents * vector_size)

        return flattened_tensor

    def preprocess_data(self, all_observations, all_observations_next, all_actions, all_actions_next, all_actions_local_next, rewards, dones, idx):

        all_observations = self.flatten(all_observations)
        all_observations_next = self.flatten(all_observations_next)
        all_actions = self.flatten(all_actions)
        all_actions_next = self.flatten(all_actions_next)
        all_actions_local_next = self.flatten(all_actions_local_next)
        reward = rewards[:,  idx].unsqueeze(-1)
        done = dones[:,  idx].unsqueeze(-1)

        return all_observations, all_observations_next, all_actions, all_actions_next, all_actions_local_next, reward, done

    def update(self, ISWeights, all_observations, all_observations_next, all_actions, all_actions_next, all_actions_local_next, rewards, dones, buffer_mode, idx):

        all_observations, all_observations_next, all_actions, all_actions_next, all_actions_local_next, reward, done = \
        self.preprocess_data(all_observations, all_observations_next, all_actions, all_actions_next, all_actions_local_next, rewards, dones, idx)

        new_priorities = None

        if buffer_mode:  # update critics with prioritized experience replay
            new_priorities = self.update_critic_PER(reward, done, all_observations, all_actions, all_observations_next, all_actions_next, ISWeights)
            self.update_actor_PER(all_observations, all_actions_local_next, ISWeights)
        else:
            self.update_critic(reward, done, all_observations, all_actions, all_observations_next, all_actions_next)
            self.update_actor(all_observations, all_actions_local_next)

        self.update_targets()

        return new_priorities

    def act(self, observations, target=False, noise_coefficient=0.0):

        if target:
            actor = self.actor_target
            actor.eval()
        else:
            actor = self.actor_local
            actor.train()

        action_values = actor(observations)
        action_noise = torch.tensor(noise_coefficient * self.noise.sample(), dtype=torch.float32, device=self.device)

        if noise_coefficient > 0.1:
            action_values += action_noise

        return action_values

    def update_actor(self, all_observations, all_local_actions):

        q_expected = self.critic_local(all_observations, all_local_actions)
        loss = - (q_expected.mean())

        self.actor_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def update_actor_PER(self, all_observations, all_local_actions, ISWeights):

        weights = torch.from_numpy(ISWeights).cuda()
        q_expected = self.critic_local(all_observations, all_local_actions)
        weighted_q_expected = q_expected * weights
        loss = - (q_expected.mean())
        # loss = - (weighted_q_expected.mean())

        self.actor_optimizer.zero_grad()
        loss.backward(retain_graph=True)
        self.actor_optimizer.step()

    def update_critic(self, rewards, dones, all_observations, all_actions, all_observations_next, all_actions_next):

        with torch.no_grad():
            q_target_next = self.critic_target(all_observations_next, all_actions_next)

        q_target = rewards + self.gamma * q_target_next.max() * (1 - dones)
        q_expected = self.critic_local(all_observations, all_actions)

        td_error = (q_target.detach() - q_expected) ** 2
        critic_loss = td_error.mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_critic_PER(self, rewards, dones, all_observations, all_actions, all_observations_next, all_actions_next, ISWeights):

        with torch.no_grad():
            q_target_next = self.critic_target(all_observations_next, all_actions_next)

        q_target = rewards + self.gamma * q_target_next.max() * (1 - dones)
        q_expected = self.critic_local(all_observations, all_actions)

        td_error = (q_target.detach() - q_expected) ** 2
        abs_error = td_error.abs()
        abs_error = abs_error.cpu().detach().numpy()

        weights = torch.from_numpy(ISWeights).cuda()
        loss = (weights * td_error).mean()

        self.critic_optimizer.zero_grad()
        loss.backward()
        self.critic_optimizer.step()

        return abs_error

    def _soft_update(self, local_model, target_model, tau):
        for target_parameters, local_parameters in zip(target_model.parameters(), local_model.parameters()):
            target_parameters.data.copy_(tau * local_parameters.data + (1.0 - tau) * target_parameters.data)

    def update_targets(self):
        self._soft_update(self.actor_local, self.actor_target, self.tau)
        self._soft_update(self.critic_local, self.critic_target, self.tau)