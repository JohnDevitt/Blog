# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

import copy
import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """

        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        for i in range(0, iterations) :
          new_vals = util.Counter()
          for state in mdp.getStates() :
            # Already the best action
            action = self.getAction(state)
            if action is not None:
              action_value = self.getQValue(state, action)
              new_vals[state] = action_value
          self.values = new_vals


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        total = 0
        transitionStatesAndProbabilities = self.mdp.getTransitionStatesAndProbs(state, action)
        for transitionStatePair in transitionStatesAndProbabilities :
          value = transitionStatePair[1] * ( self.mdp.getReward(state, action, transitionStatePair[0]) + (self.discount * self.getValue(transitionStatePair[0])) )
          total = total + value
        return total

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #----- Find the best action -----#

        # First, get all possible actions
        actions = self.mdp.getPossibleActions(state)
        if not actions :
          # Return None for the terminal state
          return None
        # Otherwise
        else:
          best_action = actions[0]
          best_value = float('-inf')
          # The best action is the action that maximises value
          for action in actions:
            action_value = self.getQValue(state, action)
            # And we find the action which maximises value
            if action_value >= best_value:
              best_value = action_value
              best_action = action
          # Just return the action, the value can be re-computed
          return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

