import pickle
import gymnasium as gym
import numpy as np
import time
from tqdm import tqdm

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
        q_values = [values[state, a] for a in actions]
        return actions[np.argmax(q_values)]

env = gym.make('FrozenLake-v1')
num_states = env.observation_space.n
num_actions = env.action_space.n

V = np.zeros(num_states)
Q = np.zeros((num_states, num_actions))
polci = np.zeros(num_states, dtype=int)
discount_factor = 0.99
num_iterations = 10000

while (True):
    # Initialize the delta variable to track the convergence of the algorithm
    delta = 0
    
    # Iterate over all states in the environment
    for s in range(num_states):
        # Initialize the max_value variable to negative infinity
        max_value = float('-inf')
        
        # Iterate over all possible actions in the environment
        for a in range(num_actions):
            # Get the next state and reward from taking the current action in the current state
            next_states = env.env.P[s][a]
            
            # Calculate the expected value of taking the current action in the current state
            expected_value = sum([p*(r + discount_factor*V[s_]) for p, s_, r, _ in next_states])
            
            # Update the Q-function for the current state-action pair
            Q[s, a] = expected_value
        
            
            # Update the max_value variable if the expected value is greater than the current max_value
            if expected_value > max_value:
                max_value = expected_value
                best_action = a
        
        # Calculate the difference between the new value and the old value for the current state
        delta = max(delta, np.abs(max_value - V[s]))
        
        # Update the value function and policy for the current state
        V[s] = max_value
        polci[s] = best_action
    
    # If the algorithm has converged, exit the loop
    if delta < 1e-9:
        break

# save the files to a file using pickle.dump
with open('saves/optimal.pkl', 'wb') as f:
    pickle.dump(Q, f)

print(polci)
print(Q)

num_episodes = 100001
Gtotal = 0.0

env = gym.make("FrozenLake-v1")
state, info = env.reset()

for i in tqdm(range(num_episodes)):

    state, info = env.reset()
    terminated = False
    truncated = False
    Gt = 0.0

    while not (terminated or truncated):
        # (i + epOffset)
        action = polci[state]
        nState, reward, terminated, truncated, info = env.step(action)
        Gt += reward
        state = nState
    Gtotal += Gt
    if (i % 10000 == 0) and i != 0:
        if((Gtotal/i )< 0.5):
            print("what?")
        print("\nSuccess Rate: " + str(Gtotal / i))

Gtotal = 0.0

state, info = env.reset()

for i in tqdm(range(num_episodes)):

    state, info = env.reset()
    terminated = False
    truncated = False
    Gt = 0.0

    while not (terminated or truncated):
        # (i + epOffset)
        action = policy(state, Q, 0.0)
        nState, reward, terminated, truncated, info = env.step(action)
        Gt += reward
        state = nState
    Gtotal += Gt
    if (i % 10000 == 0) and i != 0:
        if((Gtotal / i)< 1.0):
            print("what?")
        print("\nSuccess Rate: " + str(Gtotal / i))


