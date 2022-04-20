import numpy as np


class BrainInfo:
    def __init__(self, vector_observations=None,
                 reward=None, agents=None,
                 feedback=None, local_done=False, memory=None, vector_action=None):
        """
        Describes experience at current step of all agents linked to a brain.
        """
        self.vector_observations = vector_observations
        self.memories = memory
        self.rewards = reward
        self.agents = agents
        self.local_done = local_done
        self.feedback = feedback

        self.previous_earning = np.ones((1, self.agents.size)) * 0 #这一步和下面那个有啥区别吗?晚点看一看
        self.previous_action = np.zeros((1, self.agents.size))


class Platform(object):
    def __init__(self, config):
        self.n_agent = config['n_agent']
        self.n_task = config['n_task']
        self.n_stacked_observation = config["n_stacked_observation"]
        self.price_bound = config['price_bound']
        self.weights_bound = config['weights_bound']
        self.task_bound = config['task_budget_bound']
        self.time_bound = config['time_budget_bound']
        self.penalty_factor = config['penalty_factor']


        if config['weights'] is None:
            self.weights = np.random.randint(self.weights_bound[0], self.weights_bound[1], (self.n_agent, self.n_task))
            self.weights = self.weights/10 + 1
        else:
            self.weights = np.array(config['weights'])/10 + 1


        # 初始化参数
        self.strategy = np.zeros((self.n_agent, self.n_task))
        self.task_budget = np.random.randint(self.task_bound[0], self.task_bound[1], (self.n_task, 1))
        self.time_budget = np.random.randint(self.time_bound[0], self.time_bound[1], (self.n_agent, 1))
        self.rewards = None



    def reset(self):
        self.strategy = np.zeros((self.n_agent, self.n_task))

        # random_weight
        # self.reset_weights()

    def reset_weights(self):
        self.weights = np.random.randint(self.weights_bound[0], self.weights_bound[1], (self.n_agent, self.n_task))
        self.weights = self.weights / 10 + 1
