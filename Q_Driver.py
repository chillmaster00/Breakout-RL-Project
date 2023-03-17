


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import os

import Q_Prototype as q


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

    # Set 

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
def graph_winloss(x:list, y:list, c:str):
    winloss_x, winloss_y = get_winloss(x, y)

    plt.scatter(winloss_x, winloss_y, 1, c)

    return 0

# Runs a q learning experiment for a number of episodes
def run_test(target_episodes:int, alpha:float, gamma:float, epsilon:float, time_limit:int):
    # Starting data
    sa_values = np.zeros((16, 4))
    episode_rewards = []

    # Break up session into multiple iterations
    sa_values, episode_rewards = q.run_q_learning(sa_values, episode_rewards, target_episodes, alpha, gamma, epsilon, time_limit)

    return sa_values, episode_rewards


# Perform testing of individual changes in certain parameters
def phase_one():
    # Get optimal SA values (as calculated by Trevor Maxwell)
    try:
        with open('saves/optimal.pkl', 'rb') as f:
            optimalSA = pickle.load(f)
    except:
        optimalSA = numpy.zeros((16, 4))
        print("ERROR FINDING OPTIMAL: DEFAULTING TO ZEROS")

    # Save data
    save_fpath = "q_save_data\\Test1\\Phase1\\"
    experiment_ep_rewards = []
    
    # Default variables for TD Experiment 1
    target_episodes = 10000
    alpha = 1.0
    gamma = 1.0
    epsilon = 0.0
    time_limit = 10000

    num_runs = 30
    tests = ["Base", "a=0.5", "a=0.1", "g=0.5", "g=0.1", "e=0.5", "e=0.1"]
    rmse_results = np.zeros(len(tests))

    # Run base test
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[0] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:1], rmse_results[:1])
    save_to_file(save_fpath + "base_data.pyc", experiment_ep_rewards)

    # Run alpha-half test
    alpha = 0.5
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[1] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:2], rmse_results[:2])
    save_to_file(save_fpath + "alpha_half_data.pyc", experiment_ep_rewards)

    # Run alpha-low test
    alpha = 0.1
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[2] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:3], rmse_results[:3])
    save_to_file(save_fpath + "alpha_low_data.pyc", experiment_ep_rewards)

    # Reset alpha
    alpha = 1.0


    # Run gamma-half test
    gamma = 0.5
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[3] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:4], rmse_results[:4])
    save_to_file(save_fpath + "gamma_half_data.pyc", experiment_ep_rewards)
    
    # Run gamma-low test
    gamma = 0.1
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[4] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:5], rmse_results[:5])
    save_to_file(save_fpath + "gamma_low_data.pyc", experiment_ep_rewards)
    
    # Reset gamma
    gamma = 1.0


    # Run epsilon-half test
    epsilon = 0.5
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[5] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:6], rmse_results[:6])
    save_to_file(save_fpath + "epsilon_half_data.pyc", experiment_ep_rewards)

    # Run epsilon-low test
    epsilon = 0.1
    total_rmse = 0
    experiment_ep_rewards = []
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, time_limit)
        experiment_ep_rewards.append(episode_rewards)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[6] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:7], rmse_results[:7])
    save_to_file(save_fpath + "epsilon_low_data.pyc", experiment_ep_rewards)

    # Reset epsilon
    epsilon = 0.0

    save_bar_graph(save_fpath + "phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests, rmse_results)

    return 0


# Graphs the average episode rewards for each experiment
def phase_one_average_graph():
    # Find data to load
    load_fpath = "q_save_data\\Test1\\Phase1\\"
    
    base_data = load_from_file(load_fpath + "base_data.pyc")
    
    alpha_half_data = load_from_file(load_fpath + "alpha_half_data.pyc")
    alpha_low_data = load_from_file(load_fpath + "alpha_low_data.pyc")

    gamma_half_data = load_from_file(load_fpath + "gamma_half_data.pyc")
    gamma_low_data = load_from_file(load_fpath + "gamma_low_data.pyc")

    epsilon_half_data = load_from_file(load_fpath + "epsilon_half_data.pyc")
    epsilon_low_data = load_from_file(load_fpath + "epsilon_low_data.pyc")


    # Get the total of the sa values
    t_base_data = base_data[0]
    for i in range(1, 30):
        np.add(t_base_data, base_data[i])

        
    t_alpha_half_data = alpha_half_data[0]
    for i in range(1, 30):
        np.add(t_alpha_half_data, alpha_half_data[i])
        
    t_alpha_low_data = alpha_low_data[0]
    for i in range(1, 30):
        np.add(t_alpha_low_data, alpha_low_data[i])

        
    t_gamma_half_data = gamma_half_data[0]
    for i in range(1, 30):
        np.add(t_gamma_half_data, gamma_half_data[i])
        
    t_gamma_low_data = gamma_low_data[0]
    for i in range(1, 30):
        np.add(t_gamma_low_data, gamma_low_data[i])

        
    t_epsilon_half_data = epsilon_half_data[0]
    for i in range(1, 30):
        np.add(t_epsilon_half_data, epsilon_half_data[i])
        
    t_epsilon_low_data = epsilon_low_data[0]
    for i in range(1, 30):
        np.add(t_epsilon_low_data, epsilon_low_data[i])

    
    # Divide by number of runs to get the experiment averages
    avg_base_data = []
    for x in t_base_data:
        avg_base_data.append(x/30)


    avg_alpha_half_data = []
    for x in t_alpha_half_data:
        avg_alpha_half_data.append(x/30)

    avg_alpha_low_data = []
    for x in t_alpha_low_data:
        avg_alpha_low_data.append(x/30)


    avg_gamma_half_data = []
    for x in t_gamma_half_data:
        avg_gamma_half_data.append(x/30)

    avg_gamma_low_data = []
    for x in t_gamma_low_data:
        avg_gamma_low_data.append(x/30)


    avg_epsilon_half_data = []
    for x in t_epsilon_half_data:
        avg_epsilon_half_data.append(x/30)

    avg_epsilon_low_data = []
    for x in t_epsilon_low_data:
        avg_epsilon_low_data.append(x/30)

    # Graph the data on a single graph
    # Blue - Base
    # Green - Alpha Half
    # Red - Alpha Low
    # Cyan - Gamma Half
    # Magenta - Gamma Low
    # Yellow - Epsilon Half
    # Black(k) - Epsilon Low
    graph_winloss(range(len(avg_base_data)), base_data[0], 'b')
    graph_winloss(range(len(avg_alpha_half_data)), alpha_half_data[0], 'g')
    graph_winloss(range(len(avg_alpha_low_data)), alpha_low_data[0], 'r')
    graph_winloss(range(len(avg_gamma_half_data)), gamma_half_data[0], 'c')
    graph_winloss(range(len(avg_gamma_low_data)), gamma_low_data[0], 'm')
    graph_winloss(range(len(avg_epsilon_half_data)), epsilon_half_data[0], 'y')
    graph_winloss(range(len(avg_epsilon_low_data)), epsilon_low_data[0], 'k')
    save_graph(load_fpath + "winloss_graph.png", "Average Win/Loss Ratio over Episodes", "Episode", "Average W/L Ratio")

    
    return 0


def phase_two():
    # Get optimal SA values (as calculated by Trevor Maxwell)
    try:
        with open('saves/optimal.pkl', 'rb') as f:
            optimalSA = pickle.load(f)
    except:
        optimalSA = numpy.zeros((16, 4))
        print("ERROR FINDING OPTIMAL: DEFAULTING TO ZEROS")

    # Default variables for TD Experiment 1
    target_episodes = 10000
    alpha = 1.0
    gamma = 1.0
    epsilon = 0.0
    time_limit = 10000

    num_runs = 30
    tests = ["Base", "a=0.5", "a=0.1", "g=0.5", "g=0.1", "e=0.5", "e=0.1"]
    rmse_results = np.zeros(len(tests))

    total_rmse = 0
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, time_limit)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[0] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph("q_save_data\\Test1\\phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:1], rmse_results[:1])


    total_rmse = 0
    for i in tqdm(range(30)):
        sa_values, episode_rewards = run_test(target_episodes, alpha, gamma, epsilon, time_limit)
        total_rmse += rmsError(optimalSA, sa_values)
    rmse_results[0] = total_rmse/num_runs *100
    print(rmse_results)
    save_bar_graph("q_save_data\\Test1\\phase1_plot.png", "RMSE for Different Paramater Values", "Paramter and Value", "RMS Error (%)", tests[:1], rmse_results[:1])


    return 0

phase_one_average_graph()