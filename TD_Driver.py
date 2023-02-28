


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
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


def td_session(file_prefix, sa_values, episode_rewards, target_episodes, alpha, gamma, epsilon, time_limit):
    # File names for saving
    plot_file_name = file_prefix + "plot.png"
    sa_values_file_name = file_prefix + "sa_values.pyc"
    episode_rewards_file_name = file_prefix + "episode_rewards.pyc"
    a_file_name = file_prefix + "a.pyc"
    b_file_name = file_prefix + "b.pyc"


    # Run the TD Learning session
    sa_values, episode_rewards = td.run_td_learning(sa_values, episode_rewards, target_episodes, alpha, gamma, epsilon, time_limit)
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
    # Plot up to 100 points
    ind_increment = int(target_episodes / 100)
    ind_increment = max(1, ind_increment)
    plt.scatter(x[::ind_increment], y[::ind_increment])
    plt.plot(x,a*x+b, "r-")

    # Label graph
    plt.xlabel("Episodes")
    plt.ylabel("Rewards")
    plt.title("Reward over Episodes")

    # Save best-fit plot as a file
    plt.savefig(plot_file_name)
    plt.clf()


    # Save the calculation of a and b
    save_to_file(a_file_name, a)
    save_to_file(b_file_name, b)
    print(load_from_file(a_file_name))
    print(load_from_file(b_file_name))

    return sa_values, episode_rewards


# Variables for TD Learning session
target_episodes = 1000
alpha = 0.10
gamma = 0.90
epsilon = 0.10
time_limit = 1000

# Variables for saving session information
time = datetime.now().strftime("%Y-%m-%d_h%Hm%Ms%S")
file_path = "save_data\\" + str(target_episodes) + "\\"
file_base = "td_" + time + "_"

file_prefix = file_path + file_base

# Starting data
load_file_prefix = "save_data\\1000\\td_2023-02-28_h02m26s21_"
sa_values = load_from_file(load_file_prefix + "sa_values.pyc")
episode_rewards = load_from_file(load_file_prefix + "episode_rewards.pyc")

iterations = int(target_episodes/100)
for i in tqdm(range(0, iterations)):
    print("Executing up to " + str(i) + " episodes")
    sa_values, episode_rewards = td_session(file_prefix, sa_values, episode_rewards, (i+1) * 100, alpha, gamma, epsilon, time_limit)