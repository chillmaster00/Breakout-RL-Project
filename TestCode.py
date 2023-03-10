import gymnasium as gym
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
env = gym.make("FrozenLake-v1", render_mode="human")
observation, info = env.reset()

for _ in range(1000):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()



env.close()