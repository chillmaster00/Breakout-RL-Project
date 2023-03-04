


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import pickle
import os

import TDPrototype as td


def save_to_file(fname:str, data):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as f:
        pickle.dump(data, f)
        f.close()

    return 0

def write_to_file(fname:str, message:str):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'w') as f:
        f.write(message)
        f.close()


    return 0

def load_from_file(fname: str):
    data = 0
    try:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error loading data from " + fname)
    
    return data

def save_graph(x: list, y: list, num_pts: int, x_name: str, y_name: str, title: str, fname: str):
    a, b = np.polyfit(x, y, 1)
    ind_increment = int(len(x)/num_pts)
    ind_increment = max(1, ind_increment)

    plt.scatter(x, y)
    plt.plot(x, a*x + b, "r-")

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)


    plt.savefig(fname)
    plt.clf()
    return a, b

def td_session(file_path:str , sa_values: defaultdict, episode_rewards: list, target_episodes: int, alpha: float, gamma: float, epsilon: float, time_limit: int):
    # File names for saving
    file_prefix = file_path

    plot_file_name = file_prefix + "plot.png"
    sa_values_file_name = file_prefix + "sa_values.pyc"
    episode_rewards_file_name = file_prefix + "episode_rewards.pyc"
    info_file_name = file_prefix + "info.txt"


    # Run the TD Learning session
    sa_values, episode_rewards = td.run_td_learning(sa_values, episode_rewards, target_episodes, alpha, gamma, epsilon, time_limit)

    # Save the td training reward info
    save_to_file(sa_values_file_name, sa_values)
    save_to_file(episode_rewards_file_name, episode_rewards)


    # Fit a linear curve an estimate its growth of reward and their error.
    x = range(len(episode_rewards))[:target_episodes]
    y = episode_rewards[:target_episodes]

    # Graph the data points
    a, b = save_graph(x, y, 100, "Episodes", "Total Reward", "Rewards over Episodes", plot_file_name)

    # Save information
    message = "Y = A*X + B\n"
    message += "\tA = " + str(a) + "\n"
    message += "\tB = " + str(b) + "\n"
    message += "max = " + str(max(episode_rewards[:target_episodes])) + "\n"
    message += "target episodes = " + str(target_episodes) + "\n"
    message += "alpha = " + str(alpha) + "\n"
    message += "gamma = " + str(gamma) + "\n"
    message += "epsilon = " + str(epsilon) + "\n"
    message += "time limit = " + str(time_limit) + "\n"
    write_to_file(info_file_name, message)

    return sa_values, episode_rewards


# Variables for TD Learning session
target_episodes = 10000
alpha = 0.10
gamma = 0.90
epsilon = 0.10
time_limit = 10000

# Variables for saving session information
time = datetime.now().strftime("%Y-%m-%d_h%Hm%Ms%S")
file_path = "td_save_data\\Test2A\\"

# Starting data
sa_values = defaultdict(float)
episode_rewards = []
"""
load_file_path = file_path
sa_values = load_from_file(load_file_path + "sa_values.pyc")
episode_rewards = load_from_file(load_file_path + "episode_rewards.pyc")
"""

# Break up session into multiple iterations
ep_per_iter = 100
ep_per_iter = min(target_episodes, ep_per_iter)
iterations = int(target_episodes/ep_per_iter)
iterations = max(1, iterations)
for i in tqdm(range(0, iterations)):
    t_ep = (i+1) * ep_per_iter
    sa_values, episode_rewards = td_session(file_path, sa_values, episode_rewards, t_ep, alpha, gamma, epsilon, time_limit)