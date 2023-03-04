from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym.wrappers import TimeLimit
import pickle


    

def monte_carlo_policy_evaluation(env, gamma, num_episodes):

    # Read the saved values and training data from a file (if it exists)
    try:
        with open('saves/training.pkl', 'rb') as f:
            training_data = pickle.load(f)
    except FileNotFoundError:
        training_data = []
    try:
        with open('saves/values.pkl', 'rb') as f:
            Q = pickle.load(f)
    except FileNotFoundError:
        Q = np.zeros([env.observation_space.n, env.action_space.n])
    try:
        with open('saves/epOffset.pkl', 'rb') as f:
            epOffset = pickle.load(f)
    except FileNotFoundError:
        epOffset = 0

    # print(Q)
    # print(training_data)
    # print(epOffset)
    Gtotal = []
    N = np.zeros([env.observation_space.n, env.action_space.n])

    for i in tqdm(range(num_episodes)):
        episode = []
        epsilon_min = 0.01
        epsilon_max = 1
        epsilon_decay = 1000000

        state, info = env.reset()

        terminated = False
        truncated = False

        Gt = 0.0
        while not (terminated or truncated):
            # (i + epOffset)
            eps = max(epsilon_min, epsilon_max - (epsilon_max - epsilon_min) * (i)/ epsilon_decay)
            action = policy(state, Q, 0.2)
            next_state, reward, terminated, truncated, info = env.step(action)
            Gt += reward
            episode.append((state, action))
            state = next_state
        Gtotal.append(Gt)
        #print(episode)
        for t in range(len(episode)):
            state, action = episode[t]

            N[state, action] += 1.0
            alpha = 1.0 / N[state, action]
            Q[state, action] += alpha * (Gt - Q[state, action])

        training_data.append((i + epOffset,Gt))

            
        if (i % 10000 == 0) and i != 0:
            print("\nW/L Ratio: " + str(sum(Gtotal) / i))

    epOffset = len(training_data)
    # save the files to a file using pickle.dump
    with open('saves/values.pkl', 'wb') as f:
        pickle.dump(Q, f)
    with open('saves/training.pkl', 'wb') as f:
        pickle.dump(training_data, f)
    with open('saves/epOffset.pkl', 'wb') as f:
        pickle.dump(epOffset, f)

    return Q, training_data

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
env = gym.make("FrozenLake-v1")
# Evaluate the policy using the Monte Carlo method
values, tData = monte_carlo_policy_evaluation(env, 0.99, 2000000)

# assume that your training data is a list of (episode, reward) tuples

# extract the episode numbers and reward values from the training data
x = [data[0] for data in tData]
y = [data[1] for data in tData]

x = np.array(x, dtype=float)
# create a line plot of rewards versus episodes
plt.plot(x[::1000], y[::1000], 'bo')



# add axis labels and a title to the plot
plt.xlabel('Episodes')
plt.ylabel('Rewards')
plt.title('Training Progress')

# display the plot
plt.show()

x = np.array(x, dtype=float)
# fit a linear curve an estimate its y-values and their error.
a, b = np.polyfit(x, y, 1)
print(a)
plt.scatter(x[::1000], y[::1000])
plt.plot(x,a*x+b, "r-")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(x[::1000], y[::1000], 'o', color='tab:blue')

plt.show()
# Print the values for some example states
print('Action-Values', values)
print(a)


