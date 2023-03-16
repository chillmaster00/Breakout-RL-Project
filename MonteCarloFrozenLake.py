from collections import defaultdict
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from gym.wrappers import TimeLimit
import pickle


def rmsError(optimalQ, estimatedQ):
    # Calculate the mean squared error for each state-action pair
    mse = np.square(optimalQ - estimatedQ)
    
    # Calculate the root-mean-square error
    rms_error = np.sqrt(np.mean(mse))
    
    return rms_error

def MDPolicyEval(env, gamma, epsilon, num_episodes):

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

    # Initialize Gtotal for updating status
    Gtotal = 0.0
    for i in tqdm(range(num_episodes)):
        # initialize vars
        episode = []                # this stores the state action and reward for each episode
        state, info = env.reset()   # starts frozen lake
        terminated = False          # initialize finish flags and total reward for an episode
        truncated = False
        Gt = 0.0                    # total reward for episode
        G = 0.0                     # total reward for epidode modified by gamma


        # Continue till loop episode ends
        while not (terminated or truncated):
            action = policy(state, Q, epsilon)      # obtain action from policy
            # Take next step
            nState, reward, terminated, truncated, info = env.step(action)
            Gt += reward                            # increment total reward for episode
            episode.append((state, action, reward)) # append info to episode
            state = nState                          # increment state
        Gtotal += Gt    # increment total reward for specified time

        # Do policy evaluation for each step of the episode
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]   # Unpack episode
            G = gamma * G + reward               # Discounted reward
            N[state, action] += 1.0              # Increment state action 
            alpha = 1.0 / N[state, action]       # Set up alpha

            # Update Q function
            Q[state, action] += alpha * (G - Q[state, action])

        # append training data of rmse
        rmse = rmsError(optimalQ, Q) * 100.0
        trainingData.append((i + epOffset, rmse))

        # For showing results periodically
        if (i % 10000 == 0) and i != 0:
            rmse = rmsError(optimalQ, Q)

            print("\nRMS Error Rate: ", rmse * 100, "%")
            print("\nWin Rate: " + str(Gtotal / 10000))

            Gtotal = 0.0        # Resets total



    # Calculate the RMS error between the estimated Q-values and the optimal Q-values
    rmse = rmsError(optimalQ, Q)
    print("\nError Rate: ", rmse * 100, "%")
    epOffset = len(trainingData)    # Stores offset

    # save the files to a file using pickle.dump
    with open('saves/values.pkl', 'wb') as f:
        pickle.dump(Q, f)
    with open('saves/nVals.pkl', 'wb') as f:
        pickle.dump(N, f)
    with open('saves/training.pkl', 'wb') as f:
        pickle.dump(trainingData, f)
    with open('saves/epOffset.pkl', 'wb') as f:
        pickle.dump(epOffset, f)

    # Returns results
    return Q, trainingData

# Define the policy function
def policy(state, values, epsilon):
    # If below epsilon do random action
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1, 2, 3])
    else:   # Else do Best option
        actions = [0, 1, 2, 3]
        q_values = [values[state, a] for a in actions] 
        return actions[np.argmax(q_values)]

# Evaluate the policy using the Monte Carlo method
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")
values, tData = MDPolicyEval(env, 1.0, 0.2, 0)

# extract the episode numbers and rms values from the training data
x = [data[0] for data in tData]
y = [data[1] for data in tData]

x = np.array(x, dtype=float)    # change the episdes to float

# create a line plot of rewards versus episodes
plt.plot(x[::1000], y[::1000])  # plot every 1000 points

# add axis labels and a title to the plot
plt.xlabel('Episodes')
plt.ylabel('RMS Error (%)')
plt.title('Training Progress')
plt.show()                      # display the plot

# fit a linear curve an estimate its y-values and their error.
a, b = np.polyfit(x, y, 1)
plt.scatter(x[::1000], y[::1000])
plt.plot(x,a*x+b, "r-")
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.plot(x[::1000], y[::1000], 'o', color='tab:blue')
plt.show()

# Print the values for some example states
print('Action-Values', values)
print(a)


