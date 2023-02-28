


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import os

import TDPrototype as td


def save_to_file(fname, data):
    os.makedirs(os.path.dirname(fname), exist_ok=True)
    with open(fname, 'wb') as f:
        pickle.dump(data, f)

    return 0

def load_from_file(fname):
    data = 0
    try:
        with open(fname, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Error loading data from " + fname)
    
    return data


def td_session(file_prefix, sa_values, episode_rewards, num_episodes, alpha, gamma, epsilon, time_limit):
    # File names for saving
    plot_file_name = file_prefix + "plot.png"
    sa_values_file_name = file_prefix + "sa_values.pyc"
    episode_rewards_file_name = file_prefix + "episode_rewards.pyc"
    a_file_name = file_prefix + "a.pyc"
    b_file_name = file_prefix + "b.pyc"


    # Run the TD Learning session
    sa_values, episode_rewards = td.run_td_learning(sa_values, episode_rewards, num_episodes, alpha, gamma, epsilon, time_limit)
    print(sa_values)
    print(episode_rewards)

    # Save the td training reward info
    save_to_file(sa_values_file_name, sa_values)
    save_to_file(episode_rewards_file_name, episode_rewards)

    # Print data from the saved file
    print(load_from_file(sa_values_file_name))
    print(load_from_file(episode_rewards_file_name))


    # Fit a linear curve an estimate its growth of reward and their error.
    x = range(len(episode_rewards))
    y = episode_rewards
    a, b = np.polyfit(x, y, 1)

    # Graph the data points and line of best fit
    plt.scatter(x[::1], y[::1])
    plt.plot(x,a*x+b, "r-")

    # Label graph
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Reward over Episodes")

    # Save best-fit plot as a file
    plt.savefig(plot_file_name)
    plt.show()


    # Save the calculation of a and b
    save_to_file(a_file_name, a)
    save_to_file(b_file_name, b)
    print(load_from_file(a_file_name))
    print(load_from_file(b_file_name))

    return 0


# Variables for TD Learning session
num_episodes = 10
alpha = 0.10
gamma = 0.90
epsilon = 0.10
time_limit = 1000

# Variables for saving session information
time = datetime.now().strftime("%Y-%m-%d_h%Hm%Ms%S")
file_path = "save_data\\" + time + "\\"
file_base = "td_" + str(num_episodes) + "_"

file_prefix = file_path + file_base

# Starting data
load_file_prefix = ""
sa_values = defaultdict(float)
episode_rewards = []

td_session(file_prefix, sa_values, episode_rewards, num_episodes, alpha, gamma, epsilon, time_limit)