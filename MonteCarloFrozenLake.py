from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym.wrappers import TimeLimit
import pickle


def calculate_rms_error(optimal_q, estimated_q):
    # Calculate the mean squared error for each state-action pair
    mse = np.square(optimal_q - estimated_q)
    
    # Calculate the root-mean-square error
    rms_error = np.sqrt(np.mean(mse))
    
    return rms_error

def monte_carlo_policy_evaluation(env, gamma, epsilon, num_episodes):

    # Read the saved values and training data from a file (if it exists)
    try:
        with open('saves/training.pkl', 'rb') as f:
            trainingData = pickle.load(f)
    except FileNotFoundError:
        trainingData = []
    try:
        with open('saves/values.pkl', 'rb') as f:
            Q = pickle.load(f)
    except FileNotFoundError:
        Q = np.zeros([env.observation_space.n, env.action_space.n])
    try:
        with open('saves/nVals.pkl', 'rb') as f:
            N = pickle.load(f)
    except FileNotFoundError:
        N = np.zeros([env.observation_space.n, env.action_space.n])
    try:
        with open('saves/epOffset.pkl', 'rb') as f:
            epOffset = pickle.load(f)
    except FileNotFoundError:
        epOffset = 0
    try:
        with open('saves/optimal.pkl', 'rb') as f:
            optimalQ = pickle.load(f)
    except FileNotFoundError:
        optimalQ = np.zeros([env.observation_space.n, env.action_space.n])


    # print(Q)
    # print(training_data)
    # print(epOffset)
    Gtotal = 0.0

    for i in tqdm(range(num_episodes)):
        episode = []

        state, info = env.reset()

        terminated = False
        truncated = False

        Gt = 0.0
        while not (terminated or truncated):

            action = policy(state, Q, epsilon)
            nState, reward, terminated, truncated, info = env.step(action)
            Gt += reward
            episode.append((state, action, reward))
            state = nState
        Gtotal += Gt
        G = 0.0
        for t in reversed(range(len(episode))):
            state, action, reward= episode[t]
            G = gamma * G + reward

            N[state, action] += 1.0
            alpha = 1.0 / N[state, action]
            Q[state, action] += alpha * (G - Q[state, action])


        rms_error = calculate_rms_error(optimalQ, Q) * 100.0

        trainingData.append((i + epOffset, rms_error))

            
        if (i % 10000 == 0) and i != 0:
            rms_error = calculate_rms_error(optimalQ, Q)

            print("\nRMS Error Rate: ", rms_error * 100, "%")
            print("\nWin Rate: " + str(Gtotal / 10000))

            Gtotal = 0.0



    # Calculate the RMS error between the estimated Q-values and the optimal Q-values
    rms_error = calculate_rms_error(optimalQ, Q)
    print("\nError Rate: ", rms_error * 100, "%")

    epOffset = len(trainingData)
    # save the files to a file using pickle.dump
    with open('saves/values.pkl', 'wb') as f:
        pickle.dump(Q, f)
    with open('saves/nVals.pkl', 'wb') as f:
        pickle.dump(N, f)
    with open('saves/training.pkl', 'wb') as f:
        pickle.dump(trainingData, f)
    with open('saves/epOffset.pkl', 'wb') as f:
        pickle.dump(epOffset, f)

    return Q, trainingData

# Define the policy function
def policy(state, values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1, 2, 3])
    else:
        actions = [0, 1, 2, 3]
        q_values = [values[state, a] for a in actions]  # convert state to a tuple
        return actions[np.argmax(q_values)]


env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
# Evaluate the policy using the Monte Carlo method
values, tData = monte_carlo_policy_evaluation(env, 1.0, 0.2, 0)

# assume that your training data is a list of (episode, reward) tuples

# extract the episode numbers and reward values from the training data
x = [data[0] for data in tData]
y = [data[1] for data in tData]

x = np.array(x, dtype=float)

# create a line plot of rewards versus episodes
plt.plot(x[::1000], y[::1000])

# add axis labels and a title to the plot
plt.xlabel('Episodes')
plt.ylabel('RMS Error (%)')
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


