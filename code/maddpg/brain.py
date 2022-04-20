import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from utils.ounoise import OUNoise

from .model import ActorNetwork, CriticNetwork


class Brain:
    def __init__(self,
                 agent_count,
                 observation_size,
                 action_size,
                 actor_optim_params,
                 critic_optim_params,
                 soft_update_tau,
                 discount_gamma,
                 use_batch_norm,
                 seed,
                 actor_network_states,
                 critic_network_states,
                 device):

        self._soft_update_tau = soft_update_tau
        self._gamma = discount_gamma

        # actor networks
        self._actor_local = ActorNetwork(
            observation_size, action_size, use_batch_norm, seed
        ).to(device)

        self._actor_target = ActorNetwork(
            observation_size, action_size, use_batch_norm, seed
        ).to(device)

        # critic networks
        self._critic_local = CriticNetwork(
            observation_size * agent_count, action_size * agent_count, use_batch_norm, seed
        ).to(device)

        self._critic_target = CriticNetwork(
            observation_size * agent_count, action_size * agent_count, use_batch_norm, seed
        ).to(device)

        # optimizers
        self._actor_optimizer = optim.Adam(
            self._actor_local.parameters(),
            **actor_optim_params
        )

        self._critic_optimizer = optim.Adam(
            self._critic_local.parameters(),
            **critic_optim_params
        )

        if actor_network_states is not None:
            self._actor_local.load_state_dict(actor_network_states[0])
            self._actor_target.load_state_dict(actor_network_states[1])

        if critic_network_states is not None:
            self._critic_local.load_state_dict(critic_network_states[0])
            self._critic_target.load_state_dict(critic_network_states[1])

        self.noise = OUNoise(action_size, seed)

    def get_actor_model_states(self):
        return self._actor_local.state_dict(), self._actor_target.state_dict()

    def get_critic_model_states(self):
        return self._critic_local.state_dict(), self._critic_target.state_dict()


    # 重置噪声
    def reset(self):
        self.noise.reset()

    # 向actor网络传入观测，输出行为值
    def act(self, observation, target=False, noise=0.0, train=False):
        """
        :param observation: tensor of shape == (b, observation_size)
        :param target: true to evaluate with target
        :param noise: OU noise factor
        :param train: True for training mode else eval mode
        :return: action: tensor of shape == (b, action_size)
        """

        actor = self._actor_target if target else self._actor_local

        if train:
            actor.train()
        else:
            actor.eval()

        action_values = actor(observation)  # 向actor网络传入观测，输出行为值

        if noise > 0:
            noise = torch.tensor(noise * self.noise.sample(), dtype=observation.dtype, device=observation.device)
        else:
            noise = 0

        return action_values + noise        # 返回行为值加噪声，噪声有利于探索行为

    def act_R(self, observation, hx, cx, target=False, noise=0.0, train=False):

        actor = self._actor_target if target else self._actor_local

        if train:
            actor.train()
        else:
            actor.eval()

        action_values, (hx, cx) = actor(observation, (hx, cx))  # 向actor网络传入观测，输出行为值

        if noise > 0:
            noise = torch.tensor(noise * self.noise.sample(), dtype=observation.dtype, device=observation.device)
        else:
            noise = 0

        return action_values + noise, hx, cx     # 返回行为值加噪声，噪声有利于探索行为

    def update_actor(self, all_obs, all_pred_actions):
        """
        Actor
        :param all_obs: array of shape == (b, observation_size * n_agents)
        :param all_pred_actions: array of shape == (b, action_size * n_agents)
        :return:
        """

        actor_loss = -self._critic_local(all_obs, all_pred_actions).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self._actor_optimizer.step()

    def update_actor_PER(self, all_obs, all_pred_actions, ISWeights):

        weights = torch.from_numpy(ISWeights).cuda()
        #actor_loss = (-self._critic_local(all_obs, all_pred_actions) * weights).mean()
        actor_loss = -self._critic_local(all_obs, all_pred_actions).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self._actor_optimizer.step()

    def update_actor_R(self, all_obs, all_pred_actions, all_cell_value, all_hidden_value):


        actor_loss = -self._critic_local(all_obs, all_pred_actions, (all_cell_value, all_hidden_value)).mean()
        self._actor_optimizer.zero_grad()
        actor_loss.backward(retain_graph=True)
        self._actor_optimizer.step()

    def update_critic(self, rewards, dones, all_obs, all_actions, all_next_obs, all_next_actions):
        """
        Critic receives observation and actions of all agents as input
        :param rewards: array of shape == (b, 1)
        :param dones: array of shape == (b, 1)
        :param all_obs: array of shape == (b, n_agents, observation_size)
        :param all_actions: array of shape == (b, n_agents, action_size)
        :param all_next_obs:  array of shape == (b, n_agents, observation_size)
        :param all_next_actions: array of shape == (b, n_agents, action_size)
        """

        with torch.no_grad():
            q_target_next = self._critic_target(all_next_obs, all_next_actions)

        q_target = rewards + self._gamma * q_target_next.max() * (1 - dones)


        q_expected = self._critic_local(all_obs, all_actions)

        critic_loss = ((q_expected - q_target.detach()) ** 2).mean()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

    def update_critic_PER(self, rewards, dones, all_obs, all_actions, all_next_obs, all_next_actions, ISWeights):

        with torch.no_grad():
            q_target_next = self._critic_target(all_next_obs, all_next_actions)

        q_target = rewards + self._gamma * q_target_next.max() * (1 - dones)


        q_expected = self._critic_local(all_obs, all_actions)

        abs_error = (q_expected - q_target.detach()).abs()
        abs_error = abs_error.cpu().detach().numpy()

        # 重要性加权
        weights = torch.from_numpy(ISWeights).cuda()
        critic_loss = (weights * ((q_expected - q_target.detach()) ** 2)).mean()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

        return abs_error #更新优先级

    def update_critic_R(self, rewards, dones, all_obs, all_actions, all_next_obs, all_next_actions,
                        all_cell_value, all_next_cell_value, all_hidden_value, all_next_hidden_value):


        with torch.no_grad():
            q_target_next = self._critic_target(all_next_obs, all_next_actions, (all_next_hidden_value, all_next_cell_value))

        q_target = rewards + self._gamma * q_target_next.max() * (1 - dones)


        q_expected = self._critic_local(all_obs, all_actions, (all_hidden_value, all_cell_value))

        critic_loss = ((q_expected - q_target.detach()) ** 2).mean()

        self._critic_optimizer.zero_grad()
        critic_loss.backward()
        self._critic_optimizer.step()

    def update_targets(self):
        self._soft_update(self._actor_local, self._actor_target, self._soft_update_tau)
        self._soft_update(self._critic_local, self._critic_target, self._soft_update_tau)

    @staticmethod
    def _soft_update(local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ * θ_local + (1 - τ) * θ_target
        :param local_model: model will be copied from
        :param target_model: model will be copied to
        :param tau: interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)




