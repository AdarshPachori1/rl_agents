# # import gymnasium as gym
# import gym
# from time import sleep
# # from gym import wrappers
# before_training = "before_training.mp4"
# env = gym.make('CartPole-v0', render_mode="human")
# # env = wrappers.RecordVideo(env, 'my_awesome_dir', step_trigger = lambda x: True , video_length = 1000)
# env.reset()

# for _ in range(1000):
#     env.render()
#     sample= env.action_space.sample()
#     observation, reward, done, another, info = env.step(sample)
#     print(observation, reward, done, another, info)
#     if done: 
#         break
#     #sleep(0.1)

# env.close()

import tensorflow as tf
import numpy as np
print(np.__version__)
# val = tf.Variable(tf.ones(shape=(5, 5)), dtype=tf.float32, name="X")
# print(val)
# actions = tf.Variable(tf.ones(shape=(None, 5), dtype=tf.float32))
# K=4
# print(tf.one_hot(actions, K))


# X = tf.placeholder(tf.float32, shape=(None, D), name='X')
# G = tf.placeholder(tf.float32, shape=(None, ), name='G')
# actions = tf.placeholder(tf.float32, shape=(None, ), name='actions')
# Y_hat = X
# K=
# selected_action_values = tf.reduce_sum(
#     Y_hat * tf.one_hot(actions, K), 
#     reduction_indices = [1]
# )

# cost = tf.reduce_sum(tf.square(G-selected_action_values))