import numpy as np
import cvxpy as cvx
from collections import deque



class MobileUser(object):
    def __init__(self, agents=None, observations=None, observation_size=None, rewards=None,  dones=False):

        self.agents = agents
        self.observation_size = observation_size
        self.observations = observations
        self.rewards = rewards
        self.dones = dones

    def reset(self):
        self.rewards = np.zeros(self.agents.size)
        self.observations = np.zeros((self.agents.size, self.observation_size))
        self.dones = [False for i in range(self.agents.size)]


class SensingPlatform(object):
    def __init__(self, config):
        self.num_agent = config['num_agent']
        self.num_task = config['num_task']
        self.observation_history_length = config["observation_history_length"]

        self.quality_weights_bound = config['random_quality_weights_bound']

        if config['quality_weights'] is not None:
            self.quality_weights = np.array(config['quality_weights']) / 10 + 1
        else:
            self.random_weights()

        # Initiate SP's allocation strategy
        self.strategy = np.zeros((self.num_agent, self.num_task))

    def reset(self):
        self.strategy = np.zeros((self.num_agent, self.num_task))
        # self.random_weights() # Randomly generate weights of data quality

    def random_weights(self):
        self.quality_weights = np.random.randint(self.quality_weights_bound[0], self.quality_weights_bound[1], (self.num_agent, self.num_task))
        self.quality_weights = self.quality_weights / 10 + 1



class Environment(object):

    def __init__(self, agent_config=None, environment_config=None, train_config=None, train_log=None):

        self.num_agent = environment_config['num_agent']
        self.num_task = environment_config['num_task']
        self.task_budget = environment_config['task_budget']
        self.time_budget = environment_config['time_budget']

        self.action_size = agent_config['action_size']
        self.observation_size = agent_config['observation_size']
        self.observation_history_length = environment_config['observation_history_length']
        self.mu_cost = agent_config['mu_cost']

        self.epsilon = train_config['epsilon']

        agent_list = []
        for i in range(self.num_agent):
            agent_list.append(i)
        agents = np.array(agent_list)
        observation_size = self.observation_history_length * self.observation_size
        self.mobile_users = MobileUser(agents=agents, observation_size=observation_size)
        self.platform = SensingPlatform(config=environment_config)

        self.train_data = dict(train_log)
        self.train_data_episode = dict(train_log)

    def reset(self):

        self.platform.reset()
        self.mobile_users.reset()

        for key in self.train_data:
            self.train_data[key] = []

        actions = np.random.random((self.num_agent, self.action_size))
        self.step(actions)

    def step(self, action=None):

        prices = np.maximum(np.array(action), 0)
        task_budgets = np.ones((self.num_task, 1)) * self.task_budget
        time_budgets = np.ones((self.num_agent, 1)) * self.time_budget
        quality_weights = self.platform.quality_weights

        task_allocation, sp_reward = self.solve_optimization_problem(prices=prices, time_budgets=time_budgets, task_budgets=task_budgets, quality_weights=quality_weights)

        allocation = np.sum(task_allocation, axis=1)

        # Update history observation sequence

        self.mobile_users.observations[:,:(self.observation_history_length - 1) * self.observation_size] = self.mobile_users.observations[:, self.observation_size:]
        self.mobile_users.observations[:, -2] = prices[:, 0]  # Append lastest price
        self.mobile_users.observations[:, -1] = allocation[:]  # Append lastest allocated time

        mu_rewards = np.log(np.maximum(1 + np.squeeze(prices) * allocation - self.mu_cost * allocation, self.epsilon)) / 10
        self.mobile_users.rewards = mu_rewards

        # sum payoff of MUs, TIs, and SP
        mu_payoff = np.sum(np.squeeze(prices) * allocation)
        ti_payoff = np.sum(quality_weights * task_allocation - prices * task_allocation, axis=0)
        sp_payoff = np.sum(quality_weights * task_allocation)

        # Append data records
        self.train_data["prices"].append(prices)
        self.train_data["allocation"].append(task_allocation)
        self.train_data["MU_rewards"].append(mu_rewards)
        self.train_data["MU_payoff"].append(mu_payoff)
        self.train_data["TI_payoff"].append(ti_payoff)
        self.train_data["SP_payoff"].append(sp_payoff)

    def solve_optimization_problem(self, prices, time_budgets, task_budgets, quality_weights):

        x = cvx.Variable((self.num_agent, self.num_task))
        w = np.ones((self.num_task, 1))

        objective = cvx.Maximize(cvx.sum(cvx.log(1 + cvx.sum(cvx.multiply(x, quality_weights), axis=0))))

        constraints = [x >= 0, cvx.matmul(x, w) <= time_budgets, cvx.matmul(x.T, prices) <= task_budgets]

        prob = cvx.Problem(objective, constraints)
        rewards = prob.solve(solver=cvx.SCS)
        return x.value, rewards




