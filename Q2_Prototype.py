# q_Prototype.py
#   Written by An Nguyen
#   References "MonteCarloPrototype.py" by Trevor Maxwell
# Purpose
#   Performs Sarsa(lambda) learning on Frozen Lake

# Explanation of data structures used
#   state is the player's position
#   sa_values is a dictionary indexed by (state, action)

from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from tqdm import tqdm


# Global variables
# Actions that can be taken correspond to arrow keys
#   0 - Left
#   1 - Down
#   2 - Right
#   3 - Up
action_space = [0, 1, 2, 3]
def_action_val = 0

num_states = 16
num_actions = 4


# Given the environment and the state returned
#   by .step(), return the player position
#   which is, for FrozenLake, the integer returned
def discretize_state(env, game_state):
    return game_state


# Get the best action given the state
def get_best_action(sa_values, state):
    best_action, best_action_val = 0, -1

    # For each valid action, figure out which is best
    for a in [0, 1, 2, 3]:
        curr_action_val = sa_values[state, a]

        # Determine the best action
        if curr_action_val > best_action_val:
            best_action, best_action_val = a, curr_action_val
        
    # If no action is best, pick a random action
    if best_action_val == def_action_val:
        return np.random.choice(action_space)
    return best_action


# Return a policy action according to e-greedy
def get_e_greedy_action(sa_values, state, epsilon):
    # Check for exploration and, if so, return a random action
    if np.random.rand() < epsilon:
        return np.random.choice(action_space)
    
    # Otherwise, get the best action for the state
    return get_best_action(sa_values, state)


def run_q_learning(sa_values, episode_rewards, target_episodes, alpha, gamma, epsilon, lmbda, time_limit):
    # Create the environment
    env = gym.make("FrozenLake-v1", is_slippery=True)
    env = TimeLimit(env, max_episode_steps=time_limit)

    # Run episodes of q Learning
    for episode in range(len(episode_rewards), target_episodes):
        # Variables
        episode_reward = 0
        e_traces = np.zeros((num_states, num_actions))

        # Reset the environment
        game_state, info = env.reset()

        # Prime the loop
        curr_state = game_state
        curr_action = get_e_greedy_action(sa_values, curr_state, epsilon)
        terminated = False
        truncated  = False

        # Run the episode to completion
        while not (terminated or truncated):
            # Apply action, get next state and reward
            game_state, reward, terminated, truncated, info = env.step(curr_action)
            next_state = discretize_state(env, game_state)
            episode_reward += reward

            # Get the next action
            next_action = get_e_greedy_action(sa_values, curr_state, epsilon)
            
            # Calculate update value and increment eligibility trace
            delta = reward + gamma*sa_values[(next_state, next_action)] - sa_values[(curr_state, curr_action)]
            e_traces[(curr_state, curr_action)] += 1
            
            # Update values with eligibility trace and decay
            for s in range(num_states):
                for a in range(num_actions):
                    sa_values[(s, a)] += alpha*sa_values[(s, a)]*e_traces[(s, a)]
                    e_traces[(s, a)] = gamma*lmbda*e_traces[(s, a)]

            # Increment forward
            curr_state = next_state
            curr_action = next_action

        episode_rewards.append(episode_reward)

    return sa_values, episode_rewards


