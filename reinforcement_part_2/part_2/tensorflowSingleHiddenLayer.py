
import numpy as np
import tensorflow as tf

class neural_network(object) :

    def __init__(self, number_of_features, number_of_hidden_nodes, number_of_actions, learning_rate) :
        self.session = tf.Session()
        self.observations = tf.placeholder(tf.float32, shape = [None, number_of_features])
        self.actions = tf.placeholder(tf.int32)
        self.advantages = tf.placeholder(tf.float32)

        self.neural_net = self.build_net_architecture(number_of_features, number_of_hidden_nodes, number_of_actions)
        self.neural_net_querier = self.neural_net[0, :]
        self.neural_net_updater = self.build_net_updater(self.neural_net, tf.train.AdamOptimizer(learning_rate=learning_rate))

        self.session.run(tf.global_variables_initializer())

    def build_net_architecture(self, number_of_features, number_of_hidden_nodes, number_of_actions) :
        hidden_layer_weights = tf.get_variable('hidden_weights', shape = [number_of_features, number_of_hidden_nodes])
        hidden_layer_biases = tf.get_variable('hidden_biases', shape = [number_of_hidden_nodes])
        output_layer_weights = tf.get_variable('output_weights', shape = [number_of_hidden_nodes, number_of_actions])
        output_layer_biases = tf.get_variable('output_biases', shape = [number_of_actions])

        intermediate_values = tf.tanh(tf.add(tf.matmul(self.observations, hidden_layer_weights), hidden_layer_biases))
        return tf.nn.softmax(tf.add(tf.matmul(intermediate_values, output_layer_weights), output_layer_biases))

    def build_net_updater(self, neural_net, optimiser) :
        number_of_actions = tf.shape(neural_net)[0]
        ## This just looks at all of the actions taken in the trajectory and recalculates what the probability of each action was.
        ## The value can then be used as our p(x | theta) in the next section of the code
        action_probabilities = tf.gather_nd(neural_net, tf.transpose(tf.pack([tf.range(0, number_of_actions), self.actions])))
        ## E(x) = (sum_x log_e( p(x | theta) ) * f(x))  ...  from the policy gradient theorem
        ## Multiply by -1 because we want to do gradient ascent - or negative gradient descent
        expected_rewards = tf.reduce_sum(tf.mul(tf.log(action_probabilities), self.advantages)) * -1
        ## grad_theta(E(x)) ... The gradient of the expected value of x, with respect to theta
        gradients = list(zip(tf.gradients(expected_rewards, tf.trainable_variables()), tf.trainable_variables()))
        return optimiser.apply_gradients(gradients)

    def neural_net_query(self, observation) :
        feed = {self.observations:observation}
        action_probabilities = self.session.run(self.neural_net_querier, feed_dict=feed)
        cs = np.cumsum(action_probabilities)
        return sum(cs < np.random.rand())

    def neural_net_query_deterministic(self, observation) :
        feed = {self.observations:observation}
        action_probabilities = self.session.run(self.neural_net_querier, feed_dict=feed)
        return np.argmax(action_probabilities)

    def neural_net_update(self, observations, actions, advantages):
        self.session.run(self.neural_net_updater, feed_dict={ self.observations: observations,
            self.actions: actions, self.advantages: advantages})
