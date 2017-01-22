
import gym
import numpy as np
import theano, theano.tensor as T
from lasagne.updates import sgd, apply_momentum, rmsprop, nesterov_momentum

class app(object) :

	def __init__(self, environment_string) :
		self.environment = gym.make(environment_string)

	def generate_trajectory(self, neural_network, trajectory_max_length,
	episode_max_length, discount, display=False) :

		observations, actions, returns = [], [], []
		number_of_timesteps = 0

		while(number_of_timesteps < trajectory_max_length) :

			episode = self.run_episode(neural_network, display,
			episode_max_length, discount)

			number_of_timesteps += len(episode[0])
			observations.append(episode[0])
			actions.append(episode[1])
			returns.append(episode[2])

		advantages = self.compute_advantages(returns)

		all_observations = np.concatenate([observation for observation in observations])
		all_actions = np.concatenate([action for action in actions])
		all_advantages = np.concatenate([advantage for advantage in advantages])
		all_returns = np.concatenate([return_list for return_list in returns])


		return [all_observations, all_actions, all_advantages, all_returns]

	def run_episode(self, neural_network, display, episode_max_length, discount) :

		observations, rewards, actions = [], [], []
		observation = np.array(self.environment.reset())

		for _ in range(episode_max_length) :

			action = np.argmax(neural_network(observation.reshape(1, -1)))

			observation_raw, reward, done, _ = self.environment.step(action)
			observation = np.array(observation_raw)

			observations.append(observation)
			rewards.append(reward)
			actions.append(action)

			if display : self.environment.render()
			if done: break

		discounted_returns = self.sum_of_discounted_rewards(rewards, discount)
		return [np.matrix(observations), np.array(actions), discounted_returns]

	def sum_of_discounted_rewards(self, rewards, discount) :

		out = np.zeros(len(rewards), 'float64')
		out[-1] = rewards[-1]
		for i in reversed(xrange(len(rewards)-1)):
			out[i] = rewards[i] + discount*out[i+1]
		return out

	def compute_advantages(self, returns) :

		padded_returns_lists = []
		longest_episode_length = len(max(returns, key=len))

		for returns_list in returns:
			length_difference = longest_episode_length - len(returns_list)
			zeros = np.zeros(length_difference)
			padded_returns_lists.append(np.concatenate([returns_list, zeros]))

		baseline = np.mean(padded_returns_lists, axis=0)

		advantages = []
		for returns_list in returns :
			advantages.append(returns_list - baseline[:len(returns_list)])

		return advantages

	def train(self, neural_network, network_update, number_of_training_iterations,
		trajectory_max_length, episode_max_length, learning_rate, discount) :

		for trajectory_count in xrange(number_of_training_iterations) :

			trajectory = self.generate_trajectory(neural_network, trajectory_max_length,
			episode_max_length, discount)

			network_update(trajectory[0], trajectory[1], trajectory[2], learning_rate)

			if np.mean(trajectory[2]) >= 200 :
				self.run_episode(neural_network=neural_network, display=True, episode_max_length=300, discount=discount)

			print "------------------------------------------------------------------"
			print 'Trajectory number: '                   + str(trajectory_count + 1)
			print 'Average return over this trajectory: ' + str(np.mean(trajectory[3]))
			print 'Variance over this trajectory:       ' + str(np.std(trajectory[3]))
			print "------------------------------------------------------------------"


class neural_network_builder(object) :

	def __init__(self, number_of_features, number_of_actions, number_of_hidden_nodes) :

		self.number_of_features = number_of_features
		self.number_of_actions = number_of_actions
		self.number_of_hidden_nodes = number_of_hidden_nodes

		self.hidden_layer_weights = self.shared(np.random.randn(number_of_features, number_of_hidden_nodes) / np.sqrt(number_of_features))
		self.hidden_layer_biases = self.shared(np.zeros(number_of_hidden_nodes))
		self.output_layer_weights = self.shared(np.random.randn(number_of_hidden_nodes, number_of_actions) / np.sqrt(number_of_actions))
		self.output_layer_biases = self.shared(np.zeros(number_of_actions))

		self.parameters = [self.hidden_layer_weights,
						   self.hidden_layer_biases,
						   self.output_layer_weights,
						   self.output_layer_biases]

	def build_neural_network(self) :
		observations        = T.fmatrix()
		actions             = T.ivector()
		advantages          = T.ivector()
		learning_rate       = T.scalar()
		number_of_timesteps = observations.shape[0]

		neural_net_query_function = T.nnet.softmax((T.tanh(observations.dot(self.hidden_layer_weights) + self.hidden_layer_biases[None,:])).dot(self.output_layer_weights) + self.output_layer_biases[None,:])

		loss = T.log(neural_net_query_function[T.arange(number_of_timesteps), actions]).dot(advantages) / number_of_timesteps

		#updates_sgd = sgd(loss, self.parameters, learning_rate=learning_rate)
		updates = nesterov_momentum(loss, self.parameters, learning_rate=learning_rate, momentum=0.9)

		network_update = theano.function([observations, actions, advantages, learning_rate], [],
			updates=updates, allow_input_downcast=True, on_unused_input='ignore')
		neural_network = theano.function([observations], neural_net_query_function, allow_input_downcast=True,
			on_unused_input='ignore')

		return network_update, neural_network

	def shared(self, arr):
		return theano.shared(arr.astype('float64'))



if __name__ == "__main__":

	environment_string = 'CartPole-v0'

	app = app(environment_string)

	# Parameter definitons
	number_of_features = len(app.environment.observation_space.high)
	number_of_actions = app.environment.action_space.n
	number_of_hidden_nodes = 20
	number_of_training_iterations = 100
	trajectory_max_length = 10000
	episode_max_length = 100
	learning_rate = 0.5
	discount      = 1

	neural_network_builder = neural_network_builder(number_of_features=number_of_features,
		number_of_actions=number_of_actions, number_of_hidden_nodes=number_of_hidden_nodes)

	network_update, neural_network = neural_network_builder.build_neural_network()

	app.train(neural_network=neural_network, network_update=network_update,
		number_of_training_iterations=number_of_training_iterations,
		trajectory_max_length=trajectory_max_length,
		episode_max_length=episode_max_length, learning_rate=learning_rate,
		discount=discount)
