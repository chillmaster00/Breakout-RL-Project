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
from tqdm import tqdm


# Constant Variables
    # Determine pixel dimensions for each grid cell
cell_height = 6
cell_width = 8

    # Determine bounds of the area of interst for the ball
start_x = 8
end_x = 152
start_y = 93
end_y = 188

    # Determine defaults used in the discretize_state method
def_pos = -1
red_threshold = 150

    # Determine what value to default to if one is not present
    # A higher default action value encourages more exploration
    #   at the start
def_action_val = 0


# Given the environment and the state returned
#   by .step(), return a tuple of the discretized
#   grid position of the ball and paddle
def discretize_state(env, state):
    # Default positions of ball and paddle
    ball_x, ball_y = def_pos, def_pos
    paddle_x = def_pos

    # Loop through the bottom half of the screen
    # and find the position of the ball
    for x in range(start_x, end_x):
        for y in range (start_y, end_y):
            paddle_color = state[y][x][0]
            if paddle_color > red_threshold:
                ball_x = int((x-start_x) / cell_width)
                ball_y = int((y-start_y) / cell_height)
                break

        if (ball_x != def_pos) or (ball_y != def_pos):
            break

    # Loop through the remaining bottom half of the screen
    #   and find the position of the paddle
    screenH, screenW, _ = env.observation_space.shape

    for x in range(start_x, end_x):
        for y in range(end_y, screenH):
            paddle_color = state[y][x][0]
            if paddle_color > red_threshold:
                paddle_x = int((x-start_x) / cell_width)
                break
        
        if paddle_x != def_pos:
            break

    # Return the state as 
    return (ball_x, ball_y, paddle_x)


# Get the state-action value, returning the
#   
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
    for a in [0, 2, 3]:
        curr_action_val = get_sa_value(sa_values, state, a)

        # Determine the best action
        if curr_action_val > best_action_val:
            best_action, best_action_val = a, curr_action_val
            
    return best_action


# Return a policy action according to e-greedy
def get_e_greedy_action(sa_values, state, epsilon):
    # Check for exploration and, if so, return a random action
    if np.random.rand() < epsilon:
        return np.random.choice([0, 2, 3])
    
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



def run_td_learning(num_episodes, alpha, gamma, epsilon):
    # Variables
    sa_values = defaultdict(float)
    
    # Create the environment
    env = gym.make("ALE/Breakout-v5")

    # Run episodes of TD Learning
    for episode in tqdm(range(num_episodes)):
        # Reset the environment
        env.reset()

        # Prime the loop
        terminated = False
        truncated  = False

        # Start game by launching ball (Action 1)
        game_state, reward, terminated, truncated, info = env.step(1)
        curr_state = discretize_state(game_state)
        next_action = get_e_greedy_action(sa_values, curr_state, epsilon)

        # Get lives
        lives = info['lives']

        # Run the episode to completion
        while not (terminated or truncated):
            # Get the next game state
            game_state, reward, terminated, truncated, info = env.step(next_action)
            next_state = discretize_state(game_state)

            # Update the state-action values
            update_state_action_values(sa_values, alpha, gamma, curr_state, next_action, next_state, reward)
            curr_state = next_state
            next_action = get_e_greedy_action(sa_values, curr_state, epsilon)

            # If a life is lost, the ball needs to be fired
            if info['lives'] != lives:
                game_state, reward, terminated, truncated, info = env.step(1)
                lives = info['lives']

            


    return sa_values



run_td_learning(100, 0.10, 0.90, 0.10)
