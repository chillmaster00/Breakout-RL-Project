# TDPRototype.py
#   Written by An Nguyen
#   References "MonteCarloPrototype.py" by Trevor Maxwell
# Purpose
#   Performs TD learning on Atari's Breakout 2600

from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Constant Variables
cell_height = 6
cell_width = 8

start_x = 8
end_x = 152
start_y = 93
end_y = 188

def_pos = -1
red_threshold = 150

def_action_val = 1000 # guarantees each action for each state is tried at least once


# Given the environment and the state returned
# by .step(), return a tuple of the discretized
# grid position of the ball and paddle
def discretize_state(env, state):
    # TODO, check to see if it works
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
    # and find the position of the paddle
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


# Return a policy action according to e-greedy
def get_e_greedy_action(sa_values, state, epsilon):
    # TODO, check to see if this works
    # Check for exploration and, if so, return a random action
    # Note for actions:
    #   0 - do nothing
    #   1 - fire (spawns a ball if ball is gone)
    #   2 - move paddle to the left
    #   3 - move paddle to the right
    if np.random.rand() < epsilon:
        return np.random.choice([0, 2, 3])
    
    # Pick the best action
    action_values = []
    best_action, best_action_val = 0, -1
    for a in [0, 2, 3]:
        # Variables
        sa_pair = (state, a)
        curr_action_val = -1

        # Find a value for the state-action pair
        if (sa_pair) in sa_values:
            curr_action_val = sa_values[sa_pair]
        else:
            curr_action_val = def_action_val
        
        # Determine the best action
        if curr_action_val > best_action_val:
            best_action, best_action_val = a, curr_action_val
        

    return best_action



def run_td_learning(num_episodes, gamma, epsilon):
    # TODO
    # Variables
    policy = defaultdict(int)
    sa_values = defaultdict(float)
    episodes = []
    
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

        # Get lives
        lives = info['lives']

        # Run the episode to completion
        while not (terminated or truncated):
            print()


    return 0



run_td_learning(100, 0.90, 0.10)
