# TDPRototype.py
#   Written by An Nguyen
#   References "MonteCarloPrototype.py" by Trevor Maxwell
# Purpose
#   Performs TD learning on Atari's Breakout 2600

# Explanation of data structures used
#   state is a tuple of (ball_x, ball_y, paddle_x)
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


# Given the environment and the state returned
#   by .step(), return the player position
#   which is, for FrozenLake, the state returned
def discretize_state(env, game_state):
    return game_state


# Get the state-action value, returning a
# random value if unavailable
def get_sa_value(sa_values, state, action):
    sa_pair = (state, action)
    curr_action_val = -1

    if (sa_pair) in sa_values:
        return sa_values[sa_pair]
    else:
        return def_action_val


# Get the best action given the state
def get_best_action(sa_values, state):
    best_action, best_action_val = 0, -1

    # For each valid action, figure out which is best
    for a in [0, 1, 2, 3]:
        curr_action_val = get_sa_value(sa_values, state, a)

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


# Updates the state-action pair value
def update_state_action_values(sa_values, alpha, gamma, curr_state, action, next_state, reward):
    # Variables
    sa_pair = (curr_state, action)
    td_error = 0
    td_target = 0

    # Calculate how much the increment is
    next_state_best_action = get_best_action(sa_values, next_state)

    td_target = reward
    td_target += gamma * get_sa_value(sa_values, next_state, next_state_best_action)
    
    td_error = td_target - get_sa_value(sa_values, curr_state, action)
    #   Note: can't use sa_values directly in case
    #       the sa_pair hasn't been met before

    # Update the state-action value
    sa_values[sa_pair] = get_sa_value(sa_values, curr_state, action) # to get the default action val if needed
    sa_values[sa_pair] += alpha*td_error


    return 0



def run_td_learning(sa_values, episode_rewards, target_episodes, alpha, gamma, epsilon, time_limit):
    # Create the environment
    env = gym.make("FrozenLake-v1", is_slippery=False)
    env = TimeLimit(env, max_episode_steps=time_limit)

    # Run episodes of TD Learning
    for episode in range(len(episode_rewards), target_episodes):
        # Variables
        episode_reward = 0

        # Reset the environment
        game_state, info = env.reset()

        # Prime the loop
        curr_state = game_state
        next_action = get_e_greedy_action(sa_values, curr_state, epsilon)
        terminated = False
        truncated  = False

        # Run the episode to completion
        while not (terminated or truncated):
            # Get the next game state
            game_state, reward, terminated, truncated, info = env.step(next_action)
            next_state = discretize_state(env, game_state)
            episode_reward += reward

            # Update the state-action values
            update_state_action_values(sa_values, alpha, gamma, curr_state, next_action, next_state, reward)
            curr_state = next_state
            next_action = get_e_greedy_action(sa_values, curr_state, epsilon)

        episode_rewards.append(episode_reward)

    return sa_values, episode_rewards


