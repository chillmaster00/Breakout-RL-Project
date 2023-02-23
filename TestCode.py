import gymnasium as gym
from PIL import Image
from matplotlib import pyplot as plt
env = gym.make("ALE/Breakout-v5", render_mode="human")
observation, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

# diplays the obeservation in a picture
plt.imshow(observation)
plt.show()

# Define the height and width of the screen and the lower limit of the screen we want to search
screen_height, screen_width, _ = env.observation_space.shape
bottom_half = 90


# Define the color threshold for the red pixels
red_threshold = 150

# Loop through the bottom half of the screen and find the red pixels
red_pixels = []
for row in range(bottom_half, screen_height):
    for col in range(screen_width):
        pixel_color = observation[row][col][:]
        if pixel_color[0] > red_threshold:
            red_pixels.append((row, col))

# Print the coordinates of the red pixels
print(red_pixels)

print(reward)
print(env.observation_space)
print(env.action_space)

for _ in range(1000):
    action = 1  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()



env.close()