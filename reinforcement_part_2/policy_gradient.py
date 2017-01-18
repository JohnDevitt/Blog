
import gym
import numpy as np
import theano, theano.tensor as T

class app(object) :

	def __init__(self, environment_string) :
		self.environment = gym.make(environment_string)

	def generate_trajectory(self, policy, display, max_episode_length=500, max_trajectory_length=10000) :

		trajectory = []

		while(len(trajectory_length) < max_trajectory_length) :
			episode_data = self.run_episode(policy, display, max_episode_length)
			episode_return = self.sum_of_discounted_rewards_objective_function(episode_data)
			trajectory.append({'episode_data': episode_data, 'episode_return': episode_return})

		return trajectory

	def run_episode(self, policy, display, max_episode_length=500) :

		episode_data = []
		observation = self.environment.reset()
		
		for _ in range(max_episode_length) :
			action = policy.pick_action(observation)
			observation, reward, done, _ = self.environment.step(timestep_data['action'])
			episode_data.append({'observation': observation, 'action': action, 'reward': reward})
			if display : env.render()
			if done: break

		return trajectory

	def sum_of_discounted_rewards_objective_function(self, episode_data, discount_rate=1) :
		return sum([timestep_data['reward'] * pow(discount_rate, timestep) for timestep, timestep_data in enumerate(episode_data)])

	def train(self, policy, number_of_training_iterations=1000000) :

		for episode_index in xrange(number_of_training_iterations) :
			trajectory = self.generate_trajectory(policy, False)
			policy.neural_net_update(trajectory)


class neural_network(object) :

	def __init__(self, number_of_features, number_of_actions, number_of_hidden_nodes=200) :

		self.number_of_features = number_of_features
		self.number_of_actions = number_of_actions
		self.number_of_hidden_nodes = number_of_hidden_nodes

		self.hidden_layer_weights = shared(np.random.randn(number_of_features, number_of_hidden_nodes) / np.sqrt(number_of_features))
		self.hidden_layer_biases = shared(np.zeros(number_of_hidden_nodes))
		self.output_layer_weights = shared(learning_rate * np.random.randn(number_of_hidden_nodes ,number_of_actions))
		self.output_layer_biases = shared(np.zeros(number_of_actions))

		self.parameters = [hidden_layer_weights, hidden_layer_biases, output_layer_weights, output_layer_biases]

	def neural_net_update(self, number_of_actions, actions, advantages, stepsize, rho=0.9, epsilon=1e-9) :

		loss = T.log(self.neural_net_query[number_of_actions, actions]).dot(advantages) / number_of_actions
		gradients = T.grad(loss, params)

		updates = []

		for parameter, gradient in zip(self.parameters, gradients):
			accumulated_gradient_current = np.zeros(parameter.get_value(borrow=True).shape, dtype=parameter.dtype)
			accumulated_gradient_new = rho * accumulated_gradient_current + (1 - rho) * gradient ** 2
			updates.append((accumulated_gradient_current, accumulated_gradient_new))
			updates.append((parameter, parameter + (stepsize * gradient / T.sqrt(accumulated_gradient_current + epsilon))))
		return updates

	def neural_net_query(self, observation) :
		hidden_layer_values = T.tanh(observation.dot(self.hidden_layer_weights) + self.hidden_layer_biases[None,:])
		output_layer_values = hidden_layer_values.dot(output_layer_weights) + output_layer_biases[None,:]
		return T.nnet.softmax(output_layer_values)

	def take_action(self, observation) :
		action_probabilities = self.neural_net_query(observation)
		return action_probabilities.index(max(action_probabilities))

		

if __name__ == "__main__":
	
	environment_string = 'CartPole-v0'
	learning_rate = 1e-4


	app = app(environment_string)
	number_of_actions = app.environment.action_space.n
	number_of_features = len(app.environment.observation_space.high)

	policy = policy(learning_rate, number_of_actions, number_of_features)
	app.train(policy)

