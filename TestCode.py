import gymnasium as gym
from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
env = gym.make("ALE/Breakout-v5", render_mode="human")
observation, info = env.reset()

# Define a mapping of real state to abstract state
# Here, we discretize the blank space between the
# wall and the paddle as an 16 row by 18 column grid
cellHeight = 6
cellWidth = 8
gridX = 18
gridY = 16

stateGrid = np.zeros((gridX, gridY))

# Define start and end of discretization space
startX = 8
startY = 93
endX = 152
endY = 188


for _ in range(100):
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    print(info['lives'])

# Define the height and width of the screen and the lower limit of the screen we want to search
screen_height, screen_width, _ = env.observation_space.shape


# Define the color threshold for the red pixels
red_threshold = 150

# Loop through the bottom half of the screen and
# find the red pixels in the ball area
red_pixels = []
for x in range(startX, endX):
    for y in range(startY, endY):
        pixel_color = observation[y][x][:]
        if pixel_color[0] > red_threshold:
            red_pixels.append((x, y))

# Calculate grid position of ball
if (len(red_pixels) > 0):
    ballX = int((red_pixels[0][0] - startX) / cellWidth)
    ballY = int((red_pixels[0][1] - startY) / cellHeight)

# Loop through the bottom half of the screen and
# find the red pixels in the paddle area
red_pixels = []
for x in range(startX, endX):
    for y in range(endY, screen_height):
        pixel_color = observation[y][x][:]
        if pixel_color[0] > red_threshold:
            red_pixels.append((x, y))

# Calculate grid position of ball
if (len(red_pixels) > 0):
    paddleX = int((red_pixels[0][0] - startX) / cellWidth)

print("Ball is in grid position: (", ballX, ", ", ballY, ")")
print("Paddle is in position ",paddleX)

# Print the coordinates of the red pixels
plt.imshow(observation)
plt.show()

print(reward)
print(env.observation_space)
print(env.action_space)

for _ in range(1000):
    action = 1  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()



env.close()