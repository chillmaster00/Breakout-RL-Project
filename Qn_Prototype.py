# q_Prototype.py
#   Written by An Nguyen
#   References "MonteCarloPrototype.py" by Trevor Maxwell
# Purpose
#   Performs Q learning on Frozen Lake

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


def run_qn_learning(sa_values, episode_rewards, target_episodes, alpha, gamma, epsilon, n_step):
    # Create the environment
    env = gym.make("FrozenLake-v1", is_slippery=True)

    # Variable
    episodal_sa_values = []

    # Run episodes of q Learning
    for episode in range(len(episode_rewards), target_episodes):
        # Variables
        episode_reward = 0
        # For any index i, reward_history[i] is
        #   generated from sa_history[i]
        sa_history = []
        reward_history = []

        # Reset the environment
        game_state, info = env.reset()

        # Prime the loop
        curr_state = game_state
        curr_action = get_e_greedy_action(sa_values, curr_state, epsilon)
        sa_history.append((curr_state, curr_action))
        terminated = False
        truncated  = False

        # Run the episode to completion
        while not (terminated or truncated):

            # Get the next game state
            game_state, reward, terminated, truncated, info = env.step(curr_action)
            reward_history.append(reward)
            next_state = discretize_state(env, game_state)
            episode_reward += reward

            # Update the state-action values
            best_action = get_best_action(sa_values, next_state)
            
            # Get the n-step return if enough state-action and rewards are buffer'd
            if len(sa_history) >= n_step:
                # Total return should be:
                #   q^(n) = R_(t+1) + gamma*R_(t+2) + ... + gamma^(n-1)*R_(t+n) + gamma^n * Q(S_(t+n), A_(t+n)) - Q(S, A)
                t_return = 0
                for n in range(n_step):
                    t_return += pow(gamma, n)*reward_history[n] # for n=1,...,n_step; add gamma^n * R_(t+n+1) 
                t_return += pow(gamma, n_step)*sa_values[(next_state, best_action)] # add sa value from latest state
                t_return -= sa_values[sa_history[0]] # remove original for error

                # Update the sa value
                sa_values[sa_history[0]] += alpha * t_return

                # Remove updated sa pair and reward from history
                sa_history.pop(0)
                reward_history.pop(0)

            # Get state/action for next iteration
            curr_state = next_state
            curr_action = get_e_greedy_action(sa_values, curr_state, epsilon)
            sa_history.append((curr_state, curr_action))

        episode_rewards.append(episode_reward)
        episodal_sa_values.append(sa_values.copy())

    return sa_values, episode_rewards, episodal_sa_values


