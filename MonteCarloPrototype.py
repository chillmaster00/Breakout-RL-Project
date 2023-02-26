from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym.wrappers import TimeLimit
import pickle


def abstract_state(state):
    # Define a mapping of real state to abstract state
    # Here, we discretize the blank space between the
    # wall and the paddle as an 16 row by 18 column grid
    cellHeight = 3
    cellWidth = 4

    # Define start and end of discretization space
    startX = 8
    startY = 93
    endX = 152
    endY = 188

    ballX = -1
    ballY = -1
    paddleX = -1

    # Define the height and width of the screen and the lower limit of the screen we want to search
    screen_height, screen_width, _ = env.observation_space.shape


    # Define the color threshold for the red pixels
    red_threshold = 150
    # Loop through the bottom half of the screen and
    # find the red pixels in the ball area
    for x in range(startX, endX):
        for y in range(startY, endY):
            pixel_color = state[y][x][0]
            if pixel_color > red_threshold:
                ballX = int((x - startX) / cellWidth)
                ballY = int((y - startY) / cellHeight)
                break
        if (ballX != -1) or (ballY != -1):
            break      


    # Loop through the bottom half of the screen and
    # find the red pixels in the paddle area
    red_pixels = []
    for x in range(startX, endX):
        for y in range(endY, screen_height):
            pixel_color = state[y][x][0]
            if pixel_color > red_threshold:
                paddleX = int((x - startX) / cellWidth)
                break
        if paddleX != -1:
            break

    # Calculate grid position of ball
    if (len(red_pixels) > 0):
        paddleX = int((red_pixels[0][0] - startX) / cellWidth)

    # print("Ball is in grid position: (", ballX, ", ", ballY, ")")
    # print("Paddle is in position ",paddleX)

    abstract_state = ballX, ballY, paddleX 

    return abstract_state

    

def monte_carlo_policy_evaluation(env, gamma, num_episodes):

    # Read the saved values and training data from a file (if it exists)
    try:
        with open('training.pkl', 'rb') as f:
            training_data = pickle.load(f)
    except FileNotFoundError:
        training_data = []
    try:
        with open('values.pkl', 'rb') as f:
            values = pickle.load(f)
    except FileNotFoundError:
        values = defaultdict(float)
    try:
        with open('epOffset.pkl', 'rb') as f:
            epOffset = pickle.load(f)
    except FileNotFoundError:
        epOffset = 0

    # print(values)
    # print(training_data)
    # print(epOffset)

    for i in tqdm(range(num_episodes)):
        episode = []
        returns = defaultdict(list)
        epsilon_min = 0.01
        epsilon_max = 1
        epsilon_decay = 9500
        eps = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) * i / epsilon_decay)


        state = env.reset()

        terminated = False
        truncated = False

        next_state, reward, terminated, truncated, info = env.step(1)
        state = next_state
        aState = abstract_state(state)


        lives = 5
        lives = info['lives']

        while not (terminated or truncated):
            # (i + epOffset)
            eps = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) * (i)/ epsilon_decay)
            action = policy(aState, values, eps)
            if info['lives'] != lives:
                # Fire the ball with action 1
                action = 1
                lives = info['lives']

            next_state, reward, terminated, truncated, info = env.step(action)
            aState = abstract_state(state)
            episode.append((aState, action, reward))
            state = next_state
        G = 0
        tG = 0
        # print(episode)
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            tG += reward
            for t2 in range (t):
                for a in range (3):
                    if t2 != t:
                        if a != action:
                            returns[state, action].append(G)
                            values[state, action] = np.mean(returns[state, action])
        training_data.append((i + epOffset,tG))

    epOffset = len(training_data)
    # save the files to a file using pickle.dump
    with open('values.pkl', 'wb') as f:
        pickle.dump(values, f)
    with open('training.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    with open('epOffset.pkl', 'wb') as f:
        pickle.dump(epOffset, f)

    return values, training_data

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
        return np.random.choice([0, 1, 2, 3])
    else:
        actions = [0, 1, 2, 3]
        q_values = [values[state, a] for a in actions]  # convert state to a tuple
        return actions[np.argmax(q_values)]

# Create the environment    , render_mode="human"
env = gym.make("ALE/Breakout-v5")
env = TimeLimit(env, max_episode_steps=1000)
# Evaluate the policy using the Monte Carlo method
values, tData = monte_carlo_policy_evaluation(env, 0.99, 10000)

# assume that your training data is a list of (episode, reward) tuples

# extract the episode numbers and reward values from the training data
x = [data[0] for data in tData]
y = [data[1] for data in tData]

x = np.array(x, dtype=float)
# create a line plot of rewards versus episodes
plt.plot(x[::100], y[::100], 'bo')



# add axis labels and a title to the plot
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Training Progress')

# display the plot
plt.show()

x = np.array(x, dtype=float)
# fit a linear curve an estimate its y-values and their error.
a, b = np.polyfit(x, y, deg=1)

y_est = a * x + b
y_err = x.std() * np.sqrt(1/len(x) +
                          (x - x.mean())**2 / np.sum((x - x.mean())**2))

fig, ax = plt.subplots()
ax.plot(x, y_est, '-')
ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
ax.plot(x[::10], y[::10], 'o', color='tab:brown')

plt.show()
# Print the values for some example states
print('Action-Values', values)

