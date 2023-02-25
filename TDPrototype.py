from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Constant Variables
cellHeight = 6
cellWidth = 8

startX = 8
endX = 152
startY = 93
endY = 188

defaultPos = -1
redThreshold = 150


# Given the environment and the state returned
# by .step(), return a tuple of the discretized
# grid position of the ball and paddle
def discretize_state(env, state):
    # TODO, check to see if it works
    # Default positions of ball and paddle
    ballX, ballY = defaultPos, defaultPos
    paddleX = defaultPos

    # Loop through the bottom half of the screen
    # and find the position of the ball
    for x in range(startX, endX):
        for y in range (startY, endY):
            pixelColor = state[y][x][0]
            if pixelColor > redThreshold:
                ballX = int((x-startX) / cellWidth)
                ballY = int((y-startY) / cellHeight)
                break

        if (ballX != defaultPos) or (ballY != defaultPos):
            break

    # Loop through the remaining bottom half of the screen
    # and find the position of the paddle
    screenH, screenW, _ = env.observation_space.shape

    for x in range(startX, endX):
        for y in range(endY, screenH):
            pixelColor = state[y][x][0]
            if pixelColor > redThreshold:
                paddleX = int((x-startX) / cellWidth)
                break
        
        if paddleX != defaultPos:
            break

    # Return the state as 
    return (ballX, ballY, paddleX)


def td_policy_evaluation():
    # TODO
    # Variables
    returns = []
    return 0

def update_policy():
    # TODO
    return 0

def run_td_learning(num_episodes, gamma, epsilon):
    # TODO
    # Variables
    policy = defaultdict(int)
    episodes = []
    
    # Create the environment
    env = gym.make("ALE/Breakout-v5")

    # 

    return 0

def run_td_learning(num_episodes, gamma):
    return run_td_learning(num_episodes, gamma, 0)

def run_td_learning(num_episodes):
    return run_td_learning(num_episodes, 1)



run_td_learning(100, 0.90, 0.10)