from collections import defaultdict
import gymnasium as gym
import numpy as np

def abstract_state(state):
    # Define a mapping of real state to abstract state
    # Here, we discretize the blank space between the
    # wall and the paddle as an 16 row by 18 column grid
    cellHeight = 6
    cellWidth = 8
    gridX = 18
    gridY = 16

    stateGrid = np.zeros((gridX, gridY))

    # Define start and end of discretization space
    startX = 8
    startY = 93
    endX = 152
    endY = 188


    # Define the height and width of the screen and the lower limit of the screen we want to search
    screen_height, screen_width, _ = env.observation_space.shape


    # Define the color threshold for the red pixels
    red_threshold = 150
    # Loop through the bottom half of the screen and
    # find the red pixels in the ball area
    red_pixels = []
    for x in range(startX, endX):
        for y in range(startY, endY):
            pixel_color = state[y][x][0]
            if pixel_color > red_threshold:
                red_pixels.append((x, y))
    ballX = 0
    ballY = 0
    paddleX = 0
    # Calculate grid position of ball
    if (len(red_pixels) > 0):
        ballX = int((red_pixels[0][0] - startX) / cellWidth)
        ballY = int((red_pixels[0][1] - startY) / cellHeight)

    # Loop through the bottom half of the screen and
    # find the red pixels in the paddle area
    red_pixels = []
    for x in range(startX, endX):
        for y in range(endY, screen_height):
            pixel_color = state[y][x][:]
            if pixel_color[0] > red_threshold:
                red_pixels.append((x, y))

    # Calculate grid position of ball
    if (len(red_pixels) > 0):
        paddleX = int((red_pixels[0][0] - startX) / cellWidth)

    print("Ball is in grid position: (", ballX, ", ", ballY, ")")
    print("Paddle is in position ",paddleX)

    abstract_state = ballX, ballY, paddleX 

    return abstract_state

    

def monte_carlo_policy_evaluation(env, gamma, num_episodes):
    returns = defaultdict(list)
    values = defaultdict(float)
    for i in range(num_episodes):
        episode = []
        state = env.reset()
        terminated = False
        truncated = False
        lives = 5
        next_state, reward, terminated, truncated, info = env.step(1)
        lives = info['lives']
        state = next_state
        while not (terminated or truncated):
            action = 2
            if info['lives'] != lives:
                # Fire the ball with action 1
                action = 1
                lives = info['lives']

            next_state, reward, terminated, truncated, info = env.step(action)
            state = abstract_state(state)
            episode.append((state, action, reward))
            state = next_state
        G = 0
        print(episode)
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            for t2 in range (t):
                if t2 != t:
                    returns[state] = [G]
                    values[state] = np.mean(returns[state])
    return values

# Define the policy function
def policy(state, values, epsilon):
    """
    A policy for the Breakout game that becomes more greedy as the value function becomes more accurate.

    :param state: The current state of the game.
    :param values: The value function for the current policy.
    :param epsilon: The exploration rate.
    :return: The action to take.
    """
    if np.random.rand() < epsilon:
        return np.random.choice([0, 2, 3])
    else:
        actions = [0, 2, 3]
        q_values = [values.get((state, a), 0) for a in actions]
        return actions[np.argmax(q_values)]

# Create the environment
env = gym.make("ALE/Breakout-v5", render_mode="human")

# Evaluate the policy using the Monte Carlo method
values = monte_carlo_policy_evaluation(env, 0.99, 10)

# Print the values for some example states
print('Value for state (10, 20, 1):', values)

