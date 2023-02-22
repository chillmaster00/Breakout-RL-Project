import gymnasium as gym
import random
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam

env = gym.make('Breakout-v4')
state_size = env.observation_space.shape
action_size = env.action_space.n

# Define Deep Q-Learning model
def build_model(state_size, action_size):
    model = Sequential()
    model.add(Conv2D(32, (8,8), strides=(4,4), activation='relu', input_shape=state_size))
    model.add(Conv2D(64, (4,4), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(action_size, activation='linear'))
    model.compile(loss='mse', optimizer=Adam(lr=0.0001))
    return model

# Define Deep Q-Learning algorithm
def deep_q_learning(env, model, episodes, batch_size, gamma, epsilon, epsilon_min, epsilon_decay):
    memory = deque(maxlen=2000)
    scores = []
    for episode in range(episodes):
        state = env.reset()
        state = np.reshape(state, state_size)
        terminated = False
        score = 0
        while not terminated:
            if np.random.rand() <= epsilon:
                action = random.randrange(action_size)
            else:
                action = np.argmax(model.predict(state))
            test = env.step(action)
            next_state, reward, truncated, terminated, info = env.step(action)
            next_state = np.reshape(next_state, state_size)
            memory.append((state, action, reward, next_state, terminated))
            state = next_state
            score += reward
            if terminated:
                scores.append(score)
                print("episode: {}/{}, score: {}, epsilon: {:.2f}".format(episode, episodes, score, epsilon))
            if len(memory) > batch_size:
                minibatch = random.sample(memory, batch_size)
                for state, action, reward, next_state, terminated in minibatch:
                    target = reward
                    if not terminated:
                        target = (reward + gamma * np.amax(model.predict(next_state)[0]))
                    target_f = model.predict(state)
                    target_f[0][action] = target
                    model.fit(state, target_f, epochs=1, verbose=0)
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
    return scores

# Train the model
episodes = 1000
batch_size = 32
gamma = 0.99
epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

model = build_model(state_size, action_size)
scores = deep_q_learning(env, model, episodes, batch_size, gamma, epsilon, epsilon_min, epsilon_decay)

# Test the model
test_episodes = 10
for episode in range(test_episodes):
    state = env.reset()
    state = np.reshape(state, state_size)
    terminated = False
    score = 0
    while not terminated:
        action = np.argmax(model.predict(state))
        next_state, reward, terminated, info = env.step(action)
        next_state = np.reshape(next_state, state_size)
        state = next_state
        score += reward
        if terminated:
            print("test episode: {}/{}, score: {}".format(episode, test_episodes, score))
