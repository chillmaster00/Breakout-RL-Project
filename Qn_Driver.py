


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

import Qn_Prototype as q


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
def run_test(target_episodes:int, alpha:float, gamma:float, epsilon:float, n_step:int):
    # Starting data
    sa_values = np.zeros((16, 4))
    episode_rewards = []

    # Run the session
    sa_values, episode_rewards, episodal_sa_values = q.run_qn_learning(sa_values, episode_rewards, target_episodes, alpha, gamma, epsilon, n_step)

    return sa_values, episode_rewards, episodal_sa_values


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
    
    # Default variables for TD Experiment 1
    target_episodes = 10000
    alpha = 0.1
    gamma = 1.0
    epsilon = 0.1
    n_step = 1

    num_runs = 30
    tests = ["Base", "n=2", "n=4", "n=8"]
    rmse_results = np.zeros(len(tests))

    # Run base test
    total_rmse = 0
    experiment_episodal_sa_values = []
    for i in tqdm(range(num_runs)):
        sa_values, episode_rewards, episodal_sa_values = run_test(target_episodes, alpha, gamma, epsilon, n_step)
        total_rmse += rmsError(optimalSA, sa_values)
        experiment_episodal_sa_values.append(episodal_sa_values)
    rmse_results[0] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramater and Value", "RMS Error (%)", tests[:1], rmse_results[:1])
    save_to_file(save_fpath + "base_data.pyc", experiment_episodal_sa_values)


    # Run n=2 test
    total_rmse = 0
    experiment_episodal_sa_values = []
    for i in tqdm(range(num_runs)):
        sa_values, episode_rewards, episodal_sa_values = run_test(target_episodes, alpha, gamma, epsilon, n_step)
        total_rmse += rmsError(optimalSA, sa_values)
        experiment_episodal_sa_values.append(episodal_sa_values)
    rmse_results[1] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramater and Value", "RMS Error (%)", tests[:2], rmse_results[:2])
    save_to_file(save_fpath + "n=2_data.pyc", experiment_episodal_sa_values)
    

    # Run n=4 test
    total_rmse = 0
    experiment_episodal_sa_values = []
    for i in tqdm(range(num_runs)):
        sa_values, episode_rewards, episodal_sa_values = run_test(target_episodes, alpha, gamma, epsilon, n_step)
        total_rmse += rmsError(optimalSA, sa_values)
        experiment_episodal_sa_values.append(episodal_sa_values)
    rmse_results[2] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramater and Value", "RMS Error (%)", tests[:3], rmse_results[:3])
    save_to_file(save_fpath + "n=4_data.pyc", experiment_episodal_sa_values)

    # Run n=8 test
    total_rmse = 0
    experiment_episodal_sa_values = []
    for i in tqdm(range(num_runs)):
        sa_values, episode_rewards, episodal_sa_values = run_test(target_episodes, alpha, gamma, epsilon, n_step)
        total_rmse += rmsError(optimalSA, sa_values)
        experiment_episodal_sa_values.append(episodal_sa_values)
    rmse_results[3] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramater and Value", "RMS Error (%)", tests[:4], rmse_results[:4])
    save_to_file(save_fpath + "n=8_data.pyc", experiment_episodal_sa_values)


    return 0


# Graphs the average rmse over episodes
def graph_avg_episodal_rmse(load_fpath:str, fname:str, num_runs:int, num_episodes:int, clr:str, lbl:str):
    # Get optimal SA values (as calculated by Trevor Maxwell)
    try:
        with open('saves/optimal.pkl', 'rb') as f:
            optimalSA = pickle.load(f)
    except:
        optimalSA = np.zeros((16, 4))
        print("ERROR FINDING OPTIMAL: DEFAULTING TO ZEROS")


    # Find data to load
    # Structure of data to load is as follows:
    #   experiment_episodal_sa_values will have 30 arrays
    #       each of those arrays represent one run's episodal_sa_values
    #   episodal_sa_values will have 10000 total arrays
    #       each of those arrays are the episode's sa_values of the time    
    data = load_from_file(load_fpath + fname)


    # Get the average rmse for each episode
    # Start with total sa values
    #   where each sa_value is set to 0
    t_episodal_sa_values = []
    for ep in range (num_episodes):
        empty_sa_values = np.zeros((16, 4))
        t_episodal_sa_values.append(empty_sa_values)
        
    # Add sa_values
    for run in range(num_runs):
        for ep in range(num_episodes):
            for s in range(16):
                for a in range(4):
                    t_episodal_sa_values[ep][(s,a)] += data[run][ep][(s,a)]
    

    # Divide sa_values to get average
    avg_episodal_sa_values = []
    for ep in range(num_episodes):
        empty_sa_values = np.zeros((16,4))
        avg_episodal_sa_values.append(empty_sa_values)
        for s in range(16):
            for a in range(4):
                avg_episodal_sa_values[ep][(s,a)] = t_episodal_sa_values[ep][(s,a)] / 30

    # Get RMSE over episodes
    rmse_over_episodes = []
    for i in range(num_episodes):
        rmse = rmsError(optimalSA, avg_episodal_sa_values[i])
        rmse_over_episodes.append(rmse*100)
        

    # Graph the data on current graph
    plt.scatter(range(10000), rmse_over_episodes, s=1, color=clr, label=lbl)

    return 0


def phase_one_scatter_graph():
    fpath = "q_save_data\\Test2\\Phase1\\"

    # Graph th edata on a single graph
    # Blue - n=1
    # Green - n=2
    # Red - n=4
    # Cyan - n=8

    # Base graph (n=1)
    graph_avg_episodal_rmse(fpath, "base_data.pyc", 30, 10000, 'b', "Base")

    # n=2
    graph_avg_episodal_rmse(fpath, "n=2_data.pyc", 30, 10000, 'g', "n=2")

    # n=4
    graph_avg_episodal_rmse(fpath, "n=4_data.pyc", 30, 10000, 'r', "n=4")

    # n=8
    graph_avg_episodal_rmse(fpath, "n=8_data.pyc", 30, 10000, 'c', "n=8")

    plt.legend()
    save_graph(fpath + "avg_episodal_rmse.png", "Average RMSE Over Episodes", "Episodes", "Average RMSError (%)")
    
phase_one_scatter_graph()