# https://deeplearningcourses.com/c/deep-reinforcement-learning-in-python
# https://www.udemy.com/deep-reinforcement-learning-in-python
from __future__ import print_function, division
from builtins import range
# Note: you may need to update your version of future
# sudo pip install -U future

import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import random

def get_action(s, w):
  return 1 if s.dot(w) > 0 else 0


def play_one_episode(env):
  observation = env.reset()
  done = False
  t = 0

  while not done and t < 100:
    t += 1
    #print(observation, params)
    #action = get_action(observation[0], params)
    action = random.randint(0, 1)
    observation, reward, done, _,  info = env.step(action)
    if done:
      break

  return t


def play_multiple_episodes(env, T, params):
  episode_lengths = np.empty(T)

  for i in range(T):
    episode_lengths[i] = play_one_episode(env, params)

  avg_length = episode_lengths.mean()
  print("avg length:", avg_length)
  return avg_length


def random_search(env):
  episode_lengths = []
  best = 0
  params = None
  for t in range(100):
    new_params = np.random.random(4)*2 - 1
    avg_length = play_multiple_episodes(env, 100, new_params)
    episode_lengths.append(avg_length)

    if avg_length > best:
      params = new_params
      best = avg_length
  return episode_lengths, params


if __name__ == '__main__':
  env = gym.make('CartPole-v0')
#   episode_lengths, params = random_search(env)
#   plt.plot(episode_lengths)
#   plt.show()

  # play a final set of episodes
  # env = wrappers.Monitor(env, 'my_awesome_dir')
  env = wrappers.RecordVideo(env, 'my_awesome_dir')
  print("***Final run with final weights***:", play_one_episode(env))