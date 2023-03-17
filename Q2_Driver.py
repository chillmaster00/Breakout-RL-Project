


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

import Q2_Prototype as q


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


def save_graph(fname: str, title:str, x_name:str, y_name:str):
    # Label the graph
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)

    # Save then clear the resulting graph
    plt.savefig(fname)
    plt.clf()

    return 0


def save_bar_graph(fname: str, title:str, x_name:str, y_name:str, x: list, y:list):
    # Find range
    min = np.min(y)
    max = np.max(y)
    avg = np.average(y)
    range = max - min
    
    # Set vertical range
    plt.ylim(avg - range, avg + range)

    plt.bar(x, y)

    # Label the graph
    plt.xlabel(x_name)
    plt.ylabel(y_name)
    plt.title(title)

    # Save and then clear the resulting graph
    plt.savefig(fname)
    plt.clf()

    return 0


# Using Trevor Maxwell's work
def rmsError(optimalQ, estimatedQ):
    # Calculate the mean squared error for each state-action pair
    mse = np.square(optimalQ - estimatedQ)
    
    # Calculate the root-mean-square error
    rms_error = np.sqrt(np.mean(mse))
    
    return rms_error


# Get winloss
def get_winloss(x:list, y:list):
    winloss_x = []
    winloss_y = []
    num_wins = 0
    num_losses = 0

    for i in x:
        if y[i] > 0.5:
            num_wins += 1
        if y[i] <= 0.5:
            num_losses += 1

        if num_losses > 0:
            winloss_x.append(i)
            winloss_y.append(num_wins/num_losses)

    return winloss_x, winloss_y


# Graph using winloss
def graph_winloss(x:list, y:list, c:str, lbl:str):
    winloss_x, winloss_y = get_winloss(x, y)

    plt.scatter(winloss_x, winloss_y, s=1, color=c, label=lbl)

    return 0

# Runs a q learning experiment for a number of episodes
def run_test(target_episodes:int, alpha:float, gamma:float, epsilon:float, lmda:float, time_limit:int):
    # Starting data
    sa_values = np.zeros((16, 4))
    episode_rewards = []

    # Break up session into multiple iterations
    sa_values, episode_rewards = q.run_q_learning(sa_values, episode_rewards, target_episodes, alpha, gamma, epsilon, lmda, time_limit)

    return sa_values, episode_rewards


# Perform testing of individual changes in certain parameters
def phase_one():
    # Get optimal SA values (as calculated by Trevor Maxwell)
    try:
        with open('saves/optimal.pkl', 'rb') as f:
            optimalSA = pickle.load(f)
    except:
        optimalSA = np.zeros((16, 4))
        print("ERROR FINDING OPTIMAL: DEFAULTING TO ZEROS")

    # Save data
    save_fpath = "q_save_data\\Test2\\Phase1\\"
    experiment_ep_rewards = []
    
    # Default variables for TD Experiment 2, using best results from experiment 1
    target_episodes = 10000
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1
    time_limit = 10000

    num_runs = 30
    tests = ["l=0.1", "l=0.2", "l=0.3", "l=0.4", "l=0.5", "l=0.6", "l=0.7", "l=0.8", "l=0.9", "l=1.0"]
    rmse_results = np.zeros(len(tests))

    # Run test
    lmda = 0.1
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, lmda, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[0] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:1], rmse_results[:1])
    save_to_file(save_fpath + "l1_data.pyc", experiment_ep_rewards)

    # Run test
    lmda = 0.2
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, lmda, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[1] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:2], rmse_results[:2])
    save_to_file(save_fpath + "l2_data.pyc", experiment_ep_rewards)

    # Run test
    lmda = 0.3
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, lmda, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[2] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:3], rmse_results[:3])
    save_to_file(save_fpath + "l3_data.pyc", experiment_ep_rewards)

    # Run test
    lmda = 0.4
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, lmda, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[3] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:4], rmse_results[:4])
    save_to_file(save_fpath + "l4_data.pyc", experiment_ep_rewards)

    # Run test
    lmda = 0.5
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, lmda, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[4] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:5], rmse_results[:5])
    save_to_file(save_fpath + "l5_data.pyc", experiment_ep_rewards)

    # Run test
    lmda = 0.6
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, lmda, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[5] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:6], rmse_results[:6])
    save_to_file(save_fpath + "l6_data.pyc", experiment_ep_rewards)

    # Run test
    lmda = 0.7
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, lmda, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[6] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:7], rmse_results[:7])
    save_to_file(save_fpath + "l7_data.pyc", experiment_ep_rewards)

    # Run test
    lmda = 0.8
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, lmda, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[7] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:8], rmse_results[:8])
    save_to_file(save_fpath + "l8_data.pyc", experiment_ep_rewards)

    # Run test
    lmda = 0.9
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, lmda, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[8] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:9], rmse_results[:9])
    save_to_file(save_fpath + "l9_data.pyc", experiment_ep_rewards)


    # Run test
    lmda = 1.0
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, lmda, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[9] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:10], rmse_results[:10])
    save_to_file(save_fpath + "l10_data.pyc", experiment_ep_rewards)

    return 0


phase_one()