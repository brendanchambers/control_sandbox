__author__ = 'bc'

import gym
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
env = gym.make('CartPole-v0')  # observation space: position (x), velocity (x_dot), angle (theta), angular velocity

N_epochs = 1
N_steps = 100
N_observ = 4

measurements = np.zeros((N_steps+1, N_observ))
#predictions = np.zeros((N_observ, N_steps)) # for trying out kalman filtering

for i_episode in range(N_epochs):
    observation = env.reset()
    measurements[0,:] = observation
    for t in range(N_steps):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        measurements[t+1,:] = observation
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break

plt.figure()
titles = ['x','x_dot','theta','theta_dot']
for i_observ in range(N_observ):
    plt.subplot(N_observ+1, 1, i_observ+1)
    plt.plot(measurements[:,i_observ])
    plt.title(titles[i_observ])
plt.show()


