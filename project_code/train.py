import numpy as np
import random
import os
import pickle
import yaml
from collections import deque
from environment import Environment
from maddpg.maddpg import MADDPG


def train(environment, environment_config, train_config, agent_config):

    num_episode = train_config['num_episode']
    max_time_steps = train_config['max_time_steps']

    noise_coefficient = train_config['noise_coefficient']
    noise_decay_rate = train_config['noise_decay_rate']

    environment.reset()
    action_size = agent_config['action_size']
    observation_size = agent_config['observation_size'] * environment_config['observation_history_length']
    maddpg = MADDPG(environment=environment, observation_size=observation_size, action_size=action_size, agent_config=agent_config, train_config=train_config)

    task_budget = environment_config['task_budget']
    time_budget = environment_config['time_budget']

    train_data_window = deque(maxlen=10)
    log_path = '/log/task=' + str(task_budget) + '-time=' + str(time_budget) + '/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    for ep in range(1, num_episode + 1):

        environment.reset()
        observation = environment.mobile_users.observations
        maddpg.reset()
        episode_sum_rewards = np.zeros(maddpg.num_agent)

        for t in range(max_time_steps):

            actions = maddpg.get_all_actions(observation, noise_coefficient=noise_coefficient)

            # Random pricing
            # actions = np.random.rand(maddpg.num_agent, 1)

            environment.step(actions)

            next_observation = environment.mobile_users.observations
            rewards = np.asarray(environment.mobile_users.rewards)
            dones = np.asarray(environment.mobile_users.dones)

            maddpg.step(observation, actions, rewards, next_observation, dones)
            observation = next_observation
            episode_sum_rewards += rewards

            if np.any(dones):
                break

        noise_coefficient *= noise_decay_rate

        episode_sum_rewards = np.min(episode_sum_rewards)
        train_data_window.append(episode_sum_rewards)

        for key in environment.train_data_episode:
            environment.train_data_episode[key].append(np.mean(environment.train_data[key], axis=0))

        print('\rEpisode {}\t Minimum rewards: {:.3f}'.format(ep, np.mean(train_data_window)))

        if ep % num_episode == 0:
            with open(log_path + str(ep) + '.pkl', 'wb') as p:
                pickle.dump(environment.train_data_episode, p)
            print('Training data has been saved to ' + log_path + str(ep) + '.pkl')

if __name__ == '__main__':

    config_path = 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    train_config = config['train_config']
    agent_config = config['agent_config']
    environment_config = config['environment_config']

    train_log = {
        "prices": [],
        "allocation": [],
        "MU_rewards": [],
        "MU_payoff": [],
        "TI_payoff": [],
        "SP_payoff": [],
    }

    random.seed(1)
    np.random.seed(1)

    environment = Environment(agent_config=agent_config, environment_config=environment_config, train_config=train_config, train_log=train_log)
    train(environment=environment, environment_config=environment_config, train_config=train_config, agent_config=agent_config)
    environment.reset()



    
    

