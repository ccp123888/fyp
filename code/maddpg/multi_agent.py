import numpy as np
import random
import torch

from utils.replaybuffer import ReplayBuffer, PrioritizedReplayBuffer
from .brain import Brain


class MultiAgent:

    def __init__(self,
                 agent_count,
                 observation_size,
                 action_size,
                 train_config,
                 agent_config,
                 seed=None,
                 actor_model_states=None,
                 critic_model_states=None,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        def create_brain(idx):
            return Brain(
                agent_count=agent_count,
                observation_size=observation_size,
                action_size=action_size,
                actor_optim_params=train_config['actor_optim_params'],
                critic_optim_params=train_config['critic_optim_params'],
                soft_update_tau=train_config['soft_update_tau'],
                discount_gamma=train_config['discount_gamma'],
                use_batch_norm=False,
                seed=seed,
                actor_network_states=actor_model_states[idx] if actor_model_states else None,
                critic_network_states=critic_model_states[idx] if critic_model_states else None,
                device=device
            )

        self.brains = [create_brain(i) for i in range(agent_count)]
        self.agent_count = agent_count
        self.observation_size = observation_size
        self.action_size = action_size
        self.train_config = train_config
        self.agent_config = agent_config
        self.device = device

        self._batch_size = train_config['mini_batch_size']
        self._update_every = train_config['update_every']
        self._buffer_mode = train_config['buffer_mode']

        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        # Replay memory
        if self._buffer_mode: # Prioritized replay buffer
            self.memory = PrioritizedReplayBuffer(agent_count, observation_size, action_size, train_config['buffer_size'], self._batch_size, device,
                                                  train_config['alpha'], train_config['beta'],
                                                  train_config['beta_increment_per_sampling'])
        else: # Random replay buffer
            self.memory = ReplayBuffer(action_size, train_config['buffer_size'], self._batch_size, device)


        self.t_step = 0

    @staticmethod
    def _flatten(tensor):
        b, n_agents, d = tensor.shape
        return tensor.view(b, n_agents * d)

    # 重置所有agent网络的噪声
    def reset(self):
        for brain in self.brains:
            brain.reset()

    def step(self, obs, actions, rewards, next_obs, dones):
        """observation and learning by replay
        :param obs: array of shape == (agent_count, observation_size)
        :param actions: array of shape == (agent_count, action_size)
        :param rewards: array of shape == (agent_count,)
        :param next_obs: list of  array of shape == (agent_count, observation_size)
        :param dones: array of shape == (agent_count,)
        """
        self.memory.add(obs, actions, rewards, next_obs, dones.astype(np.uint8))

        self.t_step = (self.t_step + 1) % self._update_every

        if self.t_step == 0:
            if len(self.memory) > self._batch_size:
                self.update()

    def act(self, obs, target=False, noise=0.0):
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0)
        with torch.no_grad():
            actions = np.vstack([a.cpu().numpy() for a in self.act_torch(obs, target, noise)])
        return actions

    # 获取actor网络最终计算出来的行为值
    def act_torch(self, obs, target, noise=0.0, train=False):
        """Act based on the given batch of observations.
        :param obs: current observation, array of shape == (batch, num_agent, num_stacked_obs*observation_size)
        :param noise: noise factor
        :param train: True for training mode else eval mode
        :return: actions for given state as per current policy.
        """
        actions = [brain.act(obs[:, i], target, noise, train) for i, brain in enumerate(self.brains)]
        actions = torch.stack(actions).transpose(1, 0)
        return actions

    def update(self):   # 下面两函数可以写到这个函数里面，但是flatten函数还是单独提出来吧，毕竟用了多次，然后这个_learn函数实在step函数里用的

        if self._buffer_mode:
            ISWeights, indexes, observations, actions, rewards, next_observations, dones = self.memory.sample()
            observations = torch.from_numpy(observations).to(self.device)
            actions = torch.from_numpy(actions).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)
            next_observations = torch.from_numpy(next_observations).to(self.device)
            dones = torch.from_numpy(dones).to(self.device)
        else:
            experiences = self.memory.sample()
            observations, actions, rewards, next_observations, dones = [torch.from_numpy(e).float().to(self.device) for e in experiences]


        # 各种一维平铺
        all_obs = self._flatten(observations)
        all_actions = self._flatten(actions)
        all_next_obs = self._flatten(next_observations)


        all_target_next_actions = self._flatten(self.act_torch(next_observations, target=True, train=False).contiguous())
        all_local_actions = self.act_torch(observations, target=False, train=True).contiguous()

        # 对每个agent进行网络更新
        for i, brain in enumerate(self.brains):

            # update critics
            if self._buffer_mode:   # update critics with prioritized experience replay
                new_priorities = brain.update_critic_PER(rewards[:, i].unsqueeze(-1), dones[:, i].unsqueeze(-1),
                                                         all_obs, all_actions, all_next_obs, all_target_next_actions,ISWeights)
                self.memory.update_priorities(indexes, new_priorities)
            else:                   # update critics
                brain.update_critic(rewards[:, i].unsqueeze(-1), dones[:, i].unsqueeze(-1), all_obs, all_actions,all_next_obs, all_target_next_actions)



            # update actors
            all_local_next_actions = all_local_actions.detach()    # 当后面进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播
            all_local_next_actions[:, i] = all_local_actions[:, i]

            all_local_next_actions = self._flatten(all_local_next_actions)

            if self._buffer_mode:   # update critics with prioritized experience replay
                brain.update_actor_PER(all_obs, all_local_next_actions, ISWeights)
            else:
                brain.update_actor(all_obs, all_local_next_actions)

            # update targets
            brain.update_targets()


#----------------4月5日完8点----------------------------------------------------------------------------------

    def step_R(self, obs, actions, rewards, next_obs, cell_value, next_cell_value, hidden_value, next_hidden_value, dones):
        """observation and learning by replay
        :param obs: array of shape == (agent_count, observation_size)
        :param actions: array of shape == (agent_count, action_size)
        :param rewards: array of shape == (agent_count,)
        :param next_obs: list of  array of shape == (agent_count, observation_size)
        :param dones: array of shape == (agent_count,)
        """
        self.memory.add_R(obs, actions, rewards, next_obs, cell_value, next_cell_value, hidden_value, next_hidden_value, dones.astype(np.uint8))

        self.t_step = (self.t_step + 1) % self._update_every

        if self.t_step == 0:
            if len(self.memory) > self._batch_size:
                self.update_R()

    def update_R(self):

        experiences = self.memory.sample()

        observations, actions, rewards, next_observations, cell_value, next_cell_value, hidden_value, next_hidden_value, dones = [torch.from_numpy(e).float().to(self.device) for e in experiences]

        all_obs = self._flatten(observations)
        all_next_obs = self._flatten(next_observations)
        all_actions = self._flatten(actions)

        all_cell_value = self._flatten(cell_value)
        all_next_cell_value = self._flatten(next_cell_value)
        all_hidden_value = self._flatten(hidden_value)
        all_next_hidden_value = self._flatten(next_hidden_value)

        all_target_next_actions = self._flatten(self.act_torch(next_observations, target=True, train=False).contiguous())
        all_local_actions = self.act_torch(observations, target=False, train=True).contiguous()

        # 对每个agent进行网络更新
        for i, brain in enumerate(self.brains):

            # update critics
            brain.update_critic_R(rewards[:, i].unsqueeze(-1), dones[:, i].unsqueeze(-1), all_obs,
                                all_actions, all_next_obs, all_target_next_actions, all_cell_value,
                                all_next_cell_value, all_hidden_value, all_next_hidden_value)


            # update actors
            all_local_actions_agent = all_local_actions.detach()    # 当后面进行反向传播时，到该调用detach()的Variable就会停止，不能再继续向前进行传播
            all_local_actions_agent[:, i] = all_local_actions[:, i]
            all_local_actions_agent = self._flatten(all_local_actions_agent)
            brain.update_actor_R(all_obs, all_local_actions_agent, all_cell_value, all_hidden_value)

            # update targets
            brain.update_targets()

    def act_R(self, obs, hx, cx, target=False, noise=0.0):
        obs = torch.from_numpy(obs).float().to(self.device).unsqueeze(0) # obs: current observation, array of shape == (batch, num_agent, num_stacked_obs*observation_size)
        hx = torch.from_numpy(hx).float().to(self.device).unsqueeze(0) # 添加一个维度用于存batch
        cx = torch.from_numpy(cx).float().to(self.device).unsqueeze(0)  # 添加一个维度用于存batch

        with torch.no_grad():
            # 4 月 5 日, 代码才改到这里------------------
            act, hout, cout = self.act_torch_R(obs, hx, cx, target, noise)

            actions = np.vstack([a.cpu().numpy() for a in act])
            houts = np.vstack([a.cpu().numpy() for a in hout])
            couts = np.vstack([a.cpu().numpy() for a in cout])

        return actions, houts, couts

    def act_torch_R(self, obs, hx, cx, target, noise=0.0, train=False):
        """Act based on the given batch of observations.
        :param obs: current observation, array of shape == (batch, num_agent, num_stacked_obs*observation_size)
        :param noise: noise factor
        :param train: True for training mode else eval mode
        :return: actions for given state as per current policy.
        """

        #actions = [brain.act(obs[:, i], target, noise, train) for i, brain in enumerate(self.brains)]
        #actions = torch.stack(actions).transpose(1, 0)

        actions = []
        houts = []
        couts = []

        for i, brain in enumerate(self.brains):
            action, hout, cout = brain.act_R(obs[:, i], hx[:, i], cx[:, i], target, noise, train)
            actions.append(action)
            houts.append(hout)
            couts.append(cout)

        actions = torch.stack(actions).transpose(1, 0)
        houts = torch.stack(houts).transpose(1, 0)
        couts = torch.stack(couts).transpose(1, 0)

        return actions, houts, couts
