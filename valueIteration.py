import numpy as np

# define the MDP
P = np.array([
    [[0.7, 0.3], [0.2, 0.8]],
    [[0.8, 0.2], [0.1, 0.9]]
])
R = np.array([
    [[10, -10], [0, 0]],
    [[1, 1], [-1, 1]]
])
gamma = 0.9
# Define the MDP parameters
# state 0 is Olympus, 1 is Delphi, 2 is Delos, 3 is Dodoni
# action 0 is fly, 1 is walk, 2 is horse
numS = 4
numA = 3
P = np.zeros((numS, numA, numS)) # Transition probabilities
R = np.zeros((numS, numA, numS)) # Rewards

# Define the transition probabilities and rewards
P[0, 0, 0] = 0.1
R[0, 0, 0] = -1
P[0, 0, 1] = 0.9
R[0, 0, 1] = 2
P[0, 1, 1] = 0.2
R[0, 1, 1] = -2
P[0, 1, 3] = 0.8
R[0, 1, 3] = 2
P[1, 0, 1] = 0.3
R[1, 0, 1] = -1
P[1, 0, 2] = 0.7
R[1, 0, 2] = 5
P[1, 0, 2] = 0.7
R[1, 0, 2] = 5
P[1, 2, 0] = 0.8
R[1, 2, 0] = 1
P[1, 2, 3] = 0.2
R[1, 2, 3] = 1
P[2, 0, 2] = 0.2
R[2, 0, 2] = -1
P[2, 0, 1] = 0.4
R[2, 0, 1] = -1
P[2, 0, 3] = 0.4
R[2, 0, 3] = -1
P[3, 0, 3] = 0.3
R[3, 0, 3] = -1
P[3, 0, 0] = 0.7
R[3, 0, 0] = 2
P[3, 2, 0] = 0.7
R[3, 2, 0] = 0
P[3, 2, 1] = 0.3
R[3, 2, 1] = 1
# initialize the value function
V = np.zeros(2)
# perform value iteration
for i in range(100):
    Q = np.zeros((3, 3))
    for s in range(numS):
        for a in range(numA):
            Q[s, a] = np.sum(P[s, a, :] * (R[s, a, :] + gamma * V))
    V_new = np.max(Q, axis=1)
    if np.allclose(V, V_new, rtol=1e-6):
        break
    V = V_new

print("Optimal value function:", V)
