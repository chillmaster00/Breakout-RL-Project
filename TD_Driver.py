


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
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
    plt.scatter(x, y)

    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)


    plt.savefig(fname)
    plt.clf()
    return 0


def td_session(file_path:str , sa_values: defaultdict, episode_rewards: list, target_episodes: int, alpha: float, gamma: float, epsilon: float, time_limit: int):
    # File names for saving
    file_prefix = file_path

    rewards_plot_fname = file_prefix + "rewards_plot.png"
    winloss_plot_fname = file_prefix + "winloss_plot.png"
    sa_values_file_name = file_prefix + "sa_values.pyc"
    episode_rewards_file_name = file_prefix + "episode_rewards.pyc"
    info_file_name = file_prefix + "info.txt"


    # Run the TD Learning session
    sa_values, episode_rewards = td.run_td_learning(sa_values, episode_rewards, target_episodes, alpha, gamma, epsilon, time_limit)

    # Save the td training reward info
    save_to_file(sa_values_file_name, sa_values)
    save_to_file(episode_rewards_file_name, episode_rewards)


    # Graph the reward over episodes
    x = range(len(episode_rewards))[:target_episodes]
    rewards_y = episode_rewards[:target_episodes]

    # Graph the data points
    save_graph(x, rewards_y, 100, "Episodes", "Total Reward", "Rewards over Episodes", rewards_plot_fname)


    # Graph the win/loss ratio over episodes
    winloss_x = []
    winloss_y = []
    num_wins = 0
    num_losses = 0

    for i in x:
        if episode_rewards[i] > 0:
            num_wins += 1
        if episode_rewards[i] <= 0:
            num_losses += 1

        # prevent divison by 0
        if num_losses > 0:
            winloss_x.append(i)
            winloss_y.append(num_wins/num_losses)
        

    save_graph(winloss_x, winloss_y, 100, "Episodes", "Win/Loss Ratio", "W/L over Episodes", winloss_plot_fname)

    # Save information
    message = "\n"
    message += "target episodes = " + str(target_episodes) + "\n"
    message += "alpha = " + str(alpha) + "\n"
    message += "gamma = " + str(gamma) + "\n"
    message += "epsilon = " + str(epsilon) + "\n"
    message += "time limit = " + str(time_limit) + "\n"
    write_to_file(info_file_name, message)

    return sa_values, episode_rewards


def run_test(target_episodes:int, alpha:float, gamma:float, epsilon:float, time_limt:int, ep_per_iter:int, save_fpath:str, load_fpath: str):
    # Variables for saving session information

    # Starting data
    sa_values = defaultdict(float)
    episode_rewards = []

    if len(load_fpath) > 0:
        sa_values = load_from_file(load_fpath + "sa_values.pyc")
        episode_rewards = load_from_file(load_fpath + "episode_rewards.pyc")


    # Break up session into multiple iterations
    ep_per_iter = min(target_episodes, ep_per_iter)
    iterations = int(target_episodes/ep_per_iter)
    iterations = max(1, iterations)
    for i in tqdm(range(0, iterations)):
        t_ep = (i+1) * ep_per_iter
        sa_values, episode_rewards = td_session(save_fpath, sa_values, episode_rewards, t_ep, alpha, gamma, epsilon, time_limit)

    return 0


# Default variables for TD Experiment 1
target_episodes = 10000
alpha = 1.0
gamma = 1.0
epsilon = 0.0
time_limit = 10000
ep_per_iter = int(target_episodes/10)
save_fpath = "td_save_data\\Test1\\base\\"
load_fpath = ""

run_test(target_episodes, alpha, gamma, epsilon, time_limit, ep_per_iter, save_fpath, load_fpath)


# Run alpha-half test
alpha = 0.5
save_fpath = "td_save_data\\Test1\\alpha-half\\"
run_test(target_episodes, alpha, gamma, epsilon, time_limit, ep_per_iter, save_fpath, load_fpath)

# Run alpha-low test
alpha = 0.1
save_fpath = "td_save_data\\Test1\\alpha-low\\"
run_test(target_episodes, alpha, gamma, epsilon, time_limit, ep_per_iter, save_fpath, load_fpath)

# Reset alpha
alpha = 1.0


# Run gamma-half test
gamma = 0.5
save_fpath = "td_save_data\\Test1\\gamma-half\\"
run_test(target_episodes, alpha, gamma, epsilon, time_limit, ep_per_iter, save_fpath, load_fpath)

# Run gamma-low test
gamma = 0.1
save_fpath = "td_save_data\\Test1\\gamma-low\\"
run_test(target_episodes, alpha, gamma, epsilon, time_limit, ep_per_iter, save_fpath, load_fpath)

# Reset gamma
gamma = 1.0


# Run epsilon-half test
epsilon = 0.5
save_fpath = "td_save_data\\Test1\\epsilon_half\\"
run_test(target_episodes, alpha, gamma, epsilon, time_limit, ep_per_iter, save_fpath, load_fpath)

# Run epsilon-low test
epsilon = 0.1
save_fpath = "td_save_data\\Test1\\epsilon_low\\"
run_test(target_episodes, alpha, gamma, epsilon, time_limit, ep_per_iter, save_fpath, load_fpath)

# Reset epsilon
epsilon = 0.0
