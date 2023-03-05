import gymnasium as gym
from gym import wrappers
import numpy as np
from tqdm import tqdm

env = gym.make("FrozenLake-v1")

Q = np.zeros([env.observation_space.n, env.action_space.n])
n_s_a = np.zeros([env.observation_space.n, env.action_space.n])

num_episodes = 1000000
epsilon = 0.2
rList = []

for i in tqdm(range(num_episodes)):
    state, info = env.reset()
    rAll = 0
    terminated = False
    truncated = False
    results_list = []
    result_sum = 0.0
    while not (terminated or truncated):
        if np.random.rand() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state, :])
        new_state, reward, terminated, truncated, _ = env.step(action)
        results_list.append((state, action))
        result_sum += reward
        state = new_state
        rAll += reward
    rList.append(rAll)

    for (state, action) in results_list:
        n_s_a[state, action] += 1.0
        alpha = 1.0 / n_s_a[state, action]
        Q[state, action] += alpha * (result_sum - Q[state, action])

    if (i % 10000 == 0) and i != 0:
        print("Success rate: " + str(sum(rList) / i))

print("Success rate: " + str(sum(rList)/num_episodes))

env.close()
