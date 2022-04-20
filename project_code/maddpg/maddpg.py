import numpy as np
import torch

from .replaybuffer import ReplayBuffer, PrioritizedReplayBuffer
from .agent import Agent


class MADDPG:

    def __init__(self, environment, observation_size, action_size, train_config, agent_config,
                 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')):

        self.num_agent = environment.num_agent

        agent_list = []
        for i in range(self.num_agent):
            agent = Agent(environment=environment, train_config=train_config, device=device)
            agent_list.append(agent)

        self.agents = agent_list
        self.observation_size = observation_size
        self.action_size = action_size

        self.train_config = train_config
        self.agent_config = agent_config

        self.update_per_steps = train_config['update_per_steps']

        self.batch_size = train_config['mini_batch_size']
        self.buffer_mode = train_config['buffer_mode']

        self.device = device

        # Replay memory
        if self.buffer_mode: # Prioritized replay buffer
            self.replaybuffer = PrioritizedReplayBuffer(self.num_agent,
                                                  observation_size,
                                                  action_size,
                                                  train_config['buffer_size'],
                                                  self.batch_size,
                                                  self.device,
                                                  train_config['alpha'],
                                                  train_config['beta'],
                                                  train_config['beta_increment_per_sampling'])
        else: # Random replay buffer
            self.replaybuffer = ReplayBuffer(self.num_agent, observation_size, action_size, train_config['buffer_size'], self.batch_size, device)


        self.t_step = 0


    def reset(self):
        for agent in self.agents:
            agent.reset_noise()

    def step(self, observations, actions, rewards, next_observations, dones):

        self.replaybuffer.add(observations, actions, rewards, next_observations, dones.astype(np.uint8))

        self.t_step = (self.t_step + 1) % self.update_per_steps

        if self.t_step == 0:
            if len(self.replaybuffer) > self.batch_size:
                self.update()

    def get_all_actions(self, all_observation, target=False, noise_coefficient=0.0):

        all_observation_tensor = torch.from_numpy(all_observation).float().to(self.device).unsqueeze(0)

        temp = []
        with torch.no_grad():
            action_tensors = self.get_actor_outputs(all_observation_tensor, target, noise_coefficient)
            for action in action_tensors:
                temp.append(action.cpu().numpy())
            actions_numpy = np.vstack(temp)

        return actions_numpy

    def get_actor_outputs(self, all_observation, target, noise_coefficient=0.0):

        actions_list = []
        for i in range(len(self.agents)):
            agent = self.agents[i]
            local_observation = all_observation[:, i]
            local_action = agent.act(local_observation, target, noise_coefficient)
            actions_list.append(local_action)

        # print(actions_list)
        actions = torch.stack(actions_list).transpose(1, 0)
        # print(actions.shape)
        # print(actions)

        return actions

    def update(self):

        indexes = np.zeros(shape=self.batch_size, dtype=np.int32)
        ISWeights = np.zeros(shape=self.batch_size, dtype=np.float32)

        if self.buffer_mode:
            ISWeights, indexes, all_observations, all_actions, rewards, all_observations_next, dones = self.replaybuffer.sample()
            all_observations = torch.from_numpy(all_observations).to(self.device)
            all_actions = torch.from_numpy(all_actions).to(self.device)
            rewards = torch.from_numpy(rewards).to(self.device)
            all_observations_next = torch.from_numpy(all_observations_next).to(self.device)
            dones = torch.from_numpy(dones).to(self.device)
        else:
            experiences = self.replaybuffer.sample()
            all_observations, all_actions, rewards, all_observations_next, dones = [torch.from_numpy(e).float().to(self.device) for e in experiences]


        # actions used to compute TD-error
        all_actions_next = self.get_actor_outputs(all_observations_next, target=True).contiguous()
        all_actions_local = self.get_actor_outputs(all_observations, target=False).contiguous()

        for i in range(len(self.agents)):
            agent = self.agents[i]

            all_actions_local_next = all_actions_local.detach()
            all_actions_local_next [:, i] = all_actions_local[:, i]
            new_priorities = agent.update(ISWeights, all_observations, all_observations_next, all_actions, all_actions_next, all_actions_local_next, rewards, dones, self.buffer_mode, i)

            if new_priorities is not None and self.buffer_mode == 1:
                self.replaybuffer.update_priorities(indexes, new_priorities)
