

import gym
import numpy as np

class app(object) :

	def __init__(self, learning_rate) :
		self.learning_rate = learning_rate

	def build_environment(self) :
		return gym.make('CartPole-v0')

	# Runs a policy, records the data in an array
	def run_episode(self, policy, env) :
		
		# Initial observation
		observation = env.reset()
		# Return variable
		episode_data = []

		# Keep an episode relatively brief
		for _ in range(400) :
			
			# Pick the action according to the current policy
			action = policy.pick_action(observation)
			# And observe how that action performs
			observation, reward, done, info = env.step(action)
			env.render()
			# Action n, Reward n + 1, Observation n + 1
			episode_data.append({'action': action, 'reward': reward, 'observation': observation})

			if done :
				break

		# Simple return the data
		return episode_data

	def train(self, env) :

		# A large number of training cycles are needed for monte carlo policy
		# gradient. This should decrease for AC methods
		for _ in range(1000000000) :

			# Record of the run
			episode_data = run_episode(policy, env)
			expected_observation = get_expected_observation(episode_data)
			state_value = get_total_returns(episode_data)

			for entry in record :
				state_value = state_value - 1
				policy.update_weights()
				


	def log_likeilood(state, action) :
		# Softmax policy update
		return 0

	def get_expected_observation(self, episode_data) :
		observations = [step['observation'] for step in episode_data]
		matrix = np.array(observations)
		return matrix.mean(0).transpose()

	def get_total_returns(self, episode_data) :
		return sum([step['reward'] for step in episode_data])

class policy(object) :

	def __init__(self) :
		# 
		self.left_action_weights = np.zeros(4)
		self.right_action_weights = np.zeros(4)

	def pick_action(self, observation) :
		left = np.matmul(self.left_action_weights, observation)
		right = np.matmul(self.right_action_weights, observation)

		if left > right :
			return 0
		else :
			return 1

	def update_weights() :
		gradient = calculate_gradient(weights, expected_value)
		return learning_rate * gradient * value

	def calculate_gradient(self, weights, expected_value) :
		return weights - expected_value


if __name__ == "__main__":
	app = app(0.2)
	env = app.build_environment()
	policy = policy()
	print "Environment Built!"
	episode_data = app.run_episode(policy, env)

	print app.get_total_returns(episode_data)



