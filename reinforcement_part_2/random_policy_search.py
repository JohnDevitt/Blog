
import gym
import numpy as np


class app(object) :

	def build_env(self) :
		return gym.make('CartPole-v0')

	def train(self, env) :
		best_reward = float('-inf')
		good_solution = np.random.rand(4) * 2 - 1
		for _ in range(10000) :
			current_solution = np.random.rand(4) * 2 - 1
			reward = self.run_episode(current_solution, env)
			if reward > best_reward:
				best_reward = reward
				good_solution = current_solution

		return good_solution

	
	def run_episode(self, weights, env) :
		observation = env.reset()
		aggregate_reward = 0
		for _ in range(10000) :
			action = 0 if np.matmul(weights, observation) < 0 else 1
			observation, reward, done, info = env.step(action)
			aggregate_reward += reward
			if done:
				break
		return aggregate_reward

	def test_solution(self, weights, env) :
		observation = env.reset()
		done = False
		while not done:
			action = 0 if np.matmul(weights, observation) < 0 else 1
			env.render()
			observation, reward, done, info = env.step(action)




if __name__ == "__main__":
	app = app()
	env = app.build_env()
	good_solution = app.train(env)
	app.test_solution(good_solution, env)