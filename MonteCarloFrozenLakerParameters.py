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

    Q = np.zeros([env.observation_space.n, env.action_space.n])

    N = np.zeros([env.observation_space.n, env.action_space.n])

    try:
        with open('saves/optimal.pkl', 'rb') as f:
            optimalQ = pickle.load(f)
    except FileNotFoundError:
        optimalQ = np.zeros([env.observation_space.n, env.action_space.n])

    # print(Q)
    # print(training_data)
    # print(epOffset)
    Gtotal = 0.0

    for i in range(num_episodes):
        episode = []
        # epsilonMin = 0.01
        # epsilonMax = 1
        # epsilonDecay = 1000000

        state, info = env.reset()

        terminated = False
        truncated = False

        Gt = 0.0
        while not (terminated or truncated):
            # (i + epOffset)
            # eps = max(epsilonMin, epsilonMax - (epsilonMax - epsilonMin) * (i)/ epsilonDecay)
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

            
        if (i % 10000 == 0) and i != 0:
            rms_error = calculate_rms_error(optimalQ, Q)

            print("\nRMS Error Rate: ", rms_error * 100, "%")
            print("\nWin Rate: " + str(Gtotal / 10000))

            Gtotal = 0.0



    # Calculate the RMS error between the estimated Q-values and the optimal Q-values
    rms_error = calculate_rms_error(optimalQ, Q)
    #print("\nError Rate: ", rms_error * 100, "%")

    # save the files to a file using pickle.dump
    with open('saves/values.pkl', 'wb') as f:
        pickle.dump(Q, f)
    with open('saves/nVals.pkl', 'wb') as f:
        pickle.dump(N, f)

    return rms_error



# Define the policy function
def policy(state, values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice([0, 1, 2, 3])
    else:
        actions = [0, 1, 2, 3]
        q_values = [values[state, a] for a in actions]  # convert state to a tuple
        return actions[np.argmax(q_values)]



# Create the environment   
env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="ansi")

numberOfAvgs = 30
numEpisodes = 10000

# Evaluate the policy using the Monte Carlo method
gamma = np.array([0.90, 0.97, 0.98, 0.99, 1.0])
gammaS = gamma.astype("str")
rmsAvg = np.zeros(len(gamma))

# try:
#     with open('saves/rms.pkl', 'rb') as f:
#         rmsAvg = pickle.load(f)
# except FileNotFoundError:
#     rmsAvg = np.zeros(len(gamma))

numberOfAvgs = 30
numEpisodes = 10000

for i in range(len(gamma)):
    rms = 0.0
    for _ in tqdm(range(numberOfAvgs)):
        rms += monte_carlo_policy_evaluation(env, gamma[i], 0.2, numEpisodes)
    rmsAvg[i] = (rms/numberOfAvgs) * 100
    print("\nRMS Error", gamma[i], rmsAvg[i], "\n")


# assume that your training data is a list of (episode, reward) tuples
fig, ax = plt.subplots()
ax.bar(gamma, rmsAvg)
plt.xlabel('Gamma')
plt.ylabel('RMS Error (%)')
plt.title('Gamma Comparison Over 10,000 Episodes')
ax.set_ylim([26.0, 32.0])
plt.show()

# Evaluate the policy using the Monte Carlo method
eps = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
epsS = eps.astype("str")
rmsAvg = np.zeros(len(eps))

for i in range(len(eps)):
    rms = 0.0
    for _ in tqdm(range(numberOfAvgs)):
        rms += monte_carlo_policy_evaluation(env, 1.0, eps[i], numEpisodes)
    rmsAvg[i] = (rms/numberOfAvgs) * 100
    print("\nRMS Error", eps[i], rmsAvg[i], "\n")


# assume that your training data is a list of (episode, reward) tuples
fig, ax = plt.subplots()
ax.bar(epsS, rmsAvg)
plt.xlabel('Epsilon')
plt.ylabel('RMS Error (%)')
plt.title('Epsilon Comparison Over 10,000 Episodes')
ax.set_ylim([26.0, 32.0])
plt.show()
