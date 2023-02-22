from collections import defaultdict
import gymnasium as gym
import numpy as np

def abstract_state(state):
    # Define a mapping of real state to abstract state
    # In this example, we create a 10x10 grid on the screen and
    # map each pixel to a grid cell
    grid_size = 10
    screen_width = 160
    screen_height = 210
    cell_width = screen_width // grid_size
    cell_height = screen_height // grid_size

    # Get the position of the ball and paddle in the real state
    ball_x, ball_y = state[0][0], state[0][1]
    paddle_x, _ = state[1][0], state[1][1]

    # Map the ball and paddle positions to grid cells
    ball_cell_x = ball_x // cell_width
    ball_cell_y = ball_y // cell_height
    paddle_cell_x = paddle_x // cell_width

    # Combine the ball and paddle positions into a single abstract state
    abstract_state = ball_cell_x * grid_size**2 + ball_cell_y * grid_size + paddle_cell_x

    return abstract_state

def monte_carlo_policy_evaluation(env, policy, gamma, num_episodes):
    returns = defaultdict(list)
    values = defaultdict(float)
    for i in range(num_episodes):
        episode = []
        state = env.reset()
        terminated = False
        truncated = False
        while not terminated or truncated:
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            state = abstract_state(state)
            episode.append((state, action, reward))
            state = next_state
            env.step(1)
        G = 0
        for t in reversed(range(len(episode))):
            state, action, reward = episode[t]
            G = gamma * G + reward
            if state not in [x[0] for x in episode[0:t]]:
                if state in returns:
                    returns[state].append(G)
                else:
                    returns[state] = [G]
                values[state] = np.mean(returns[state])
    return values

# Define the policy function
def policy(state):
    """
    A simple policy for the Breakout game.

    :param state: The current state of the game.
    :return: The action to take.
    """
    if np.random.uniform() < 0.5:
        return 2 
    else:
        return 3

# Create the environment
env = gym.make("ALE/Breakout-v5", render_mode="human")

# Evaluate the policy using the Monte Carlo method
values = monte_carlo_policy_evaluation(env, policy, 0.99, 1000)

# Print the values for some example states
print('Value for state (10, 20, 1):', values[(10, 20, 1)])
print('Value for state (50, 100, 2):', values[(50, 100, 2)])
print('Value for state (100, 150, 0):', values[(100, 150, 0)])
