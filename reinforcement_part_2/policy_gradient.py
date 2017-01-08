

import gym
import numpy as np

class app(object) :

	def build_environment(self) :
		return gym.make('CartPole-v0')

	def train(self, env) :

		learning_rate = 0.1

		for _ in range(1000000000) :

			# Record of the run
			record = run_episode(policy, env)
			
			# Rewards are undiscounted for continuous tasks
			state_value = 0
			for entry in record :
				state_value = state_value + entry['reward']

			for entry in record :
				policy.left_action_weights = learning_rate * log_likeilood(state, action) * reward
				policy.right_action_weights = learning_rate * log_likeilood(state, action) * reward
				# Better here? Or further up in the loop?
				state_value = state_value - entry['reward']



	# Runs a policy, records the data in an array
	def run_episode(self, policy, env) :
		
		run_record = []

		observation = env.reset()
		aggregate_reward = 0
		done = False
		# Keep an episode relatively brief
		for _ in 400 :
			policy.pick_action(obsevation)
			observation, reward, done, info = env.step(action)
			# State n, Action n, Reward n + 1
			run_record.append({'state': observation, 'action': action, 'reward': reward}})


	def log_likeilood(state, action) :
		# Softmax policy update
		

class policy(object) :

	def __init__() :
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


if __name__ == "__main__":
	app = app()
	env = app.build_environment()
	policy =
	best_left_policy, best_right_policy = app.train(env)
	print best_weights
