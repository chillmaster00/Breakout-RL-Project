


from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

import TDPrototype as td

# Run the TD Learning experiment
sa_values = defaultdict(float)
num_episodes = 10
alpha = 0.10
gamma = 0.90
epsilon = 0.10
time_limit = 1000

sa_values, episode_rewards = td.run_td_learning(sa_values, num_episodes, alpha, gamma, epsilon, time_limit)
print(sa_values)
print(episode_rewards)

# Time now
time = datetime.now().strftime("%Y%m%d_%H%M%S")
plot_file_name = time + "_plot_" + str(num_episodes) + ".png"
best_fit_file_name = time + "_best_fit_" + str(num_episodes) + ".png"


# Plot the rewards over episodes
x = range(num_episodes)
y = episode_rewards
plt.plot(x, y)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Reward over Episodes")

# save plot as a file
plt.savefig(plot_file_name)
plt.clf()

# fit a linear curve an estimate its growth of reward and their error.
a, b = np.polyfit(x, y, 1)
plt.scatter(x[::1], y[::1])
plt.plot(x,a*x+b, "r-")
plt.xlabel("Episodes")
plt.ylabel("Rewards")

# save best-fit plot as a file
plt.savefig(best_fit_file_name)
plt.show()

print("A = ", a)
print("B = ", b)