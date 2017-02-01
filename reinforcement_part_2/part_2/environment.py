
import numpy as np
from scipy.signal import lfilter

import gym

class environment(object) :

    def __init__(self, environment_string) :
        self.environment = gym.make(environment_string)

    def generate_trajectory(self, trajectory_size, neural_network, discount, display=False, deterministic=False) :

        observations, actions, returns = [], [], []

        for _ in xrange(trajectory_size) :
            obs, acts, rets = self.run_episode(neural_network, discount, display, deterministic)
            observations.append(obs)
            actions.append(acts)
            returns.append(rets)

        advantages = self.compute_advantages(returns)
        return np.concatenate(observations), np.concatenate(actions), np.concatenate(advantages), np.concatenate(returns)

    def run_episode(self, neural_network, discount, display, deterministic) :

        observations, actions, rewards = [], [], []
        observation_raw = self.environment.reset()
        observation = np.array(observation_raw, 'float64')
        done = False

        while not done :

            observations.append(observation)

            if deterministic :
                action = neural_network.neural_net_query_deterministic(observation.reshape(1, -1))
            else:
                action = neural_network.neural_net_query(observation.reshape(1, -1))
            observation_raw, reward, done, _ = self.environment.step(action)
            observation = np.array(observation_raw, 'float64')

            actions.append(action)
            rewards.append(reward)

            if display : self.environment.render()

        returns = self.compute_returns(rewards, discount)
        return np.array(observations, 'float64'), np.array(actions, 'float64'), np.array(returns, 'float64')

    def compute_returns(self, rewards, discount) :
        return lfilter([1],[1,-discount],rewards[::-1])[::-1]

    def compute_advantages(self, returns) :
        padded_returns_lists = []
        longest_episode_length = len(max(returns, key=len))

        for returns_list in returns:
            length_difference = longest_episode_length - len(returns_list)
            zeros = np.zeros(length_difference, 'float64')
            padded_returns_lists.append(np.concatenate([returns_list, zeros]))

        baseline = np.mean(padded_returns_lists, axis=0)

        advantages = []

        for returns_list in returns :
            advantages.append(returns_list - baseline[:len(returns_list)])

        return np.array(advantages)
