
import tensorflowSingleHiddenLayer
import environment
import numpy as np

if __name__ == "__main__" :

    environment_string = 'CartPole-v0'
    environment = environment.environment(environment_string)

    number_of_features            = len(environment.environment.observation_space.high)
    number_of_actions             = environment.environment.action_space.n
    hidden_layer_size             = 8
    learning_rate                 = 0.01

    number_of_training_iterations = 20000
    trajectory_size               = 20
    discount                      = 1

    neural_net = tensorflowSingleHiddenLayer.neural_network(number_of_features, hidden_layer_size, number_of_actions, learning_rate)
    average_returns = []

    for _ in xrange(number_of_training_iterations) :
        observations, actions, advantages, returns = environment.generate_trajectory(trajectory_size, neural_net, discount)
        neural_net.neural_net_update(observations, actions, advantages)

        print 'Average return over this trajectory: ' + str(np.mean(returns))
        average_returns.append(np.mean(returns))

        target = open('returns.txt', 'w')
        target.write(str(average_returns))
        target.close()

        if(np.mean(returns) > 350) :
            environment.generate_trajectory(1, neural_net, discount, display=True, deterministic=True)
