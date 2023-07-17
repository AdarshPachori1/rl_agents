import gym
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime

class HiddenLayer:
    def __init__(self, M1, M2, f=tf.nn.tanh, use_bias=True):
        self.W = tf.Variable(tf.random.normal(shape=(M1, M2)))
        self.params = [self.W]
        self.use_bias = use_bias
        if use_bias:
            self.b = tf.Variable(np.zeros(M2).astype(np.float32))
            self.params.append(self.b)
        self.f = f
    
    def forward(self, X):
        if self.use_bias:
            return self.f(tf.linalg.matmul(X, self.W)+self.b)
        return self.f(tf.linalg.matmul(X, self.W))
    
class DQN:
    def __init__(self, D, K, hidden_layer_sizes, gamma, max_experiences=10000, min_experiences=100, batch_sz=32):
        # num of inputs = d, num of outputs = k, 
        self.K = K
        self.layers =[]
        M1=D
        for M2 in hidden_layer_sizes:
            layer = HiddenLayer(M1, M2)
            self.layers.append(layer)
            M1=M2

        layer = HiddenLayer(M1, K, lambda x:x)
        self.layers.append(layer)

        self.params = []
        for layer in self.layers:
            self.params+=layer.params
        
        num_samples_batch = 1
        self.X = tf.Variable(tf.ones(shape=(num_samples_batch, D), ), dtype=tf.float32, name = "X")
        self.G = tf.Variable(tf.ones(shape=(num_samples_batch,)), dtype=tf.float32, name = "G")
        self.actions = tf.Variable(tf.ones(shape=(K,), ), dtype=tf.float32, name = "actions")

        Z = self.X
        for layer in self.layers:
            Z=layer.forward(Z)
        Y_hat = Z
        self.predict_op = Y_hat

        selected_action_values = tf.reduce_sum(
            Y_hat * tf.one_hot(self.actions.astype(np.int32), K), 
            reduction_indices = [1]
        )
        print("shape:", selected_action_values.shape)
        cost = tf.reduce_sum(tf.square(self.G-selected_action_values))

        self.train_op = tf.train.AdagradOptimizer(10e-3).minimize(cost)

        #replay memory
        self.experience = {'s': [], 'a':[], 'r':[], 's2':[]}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences
        self.batch_sz = batch_sz
        self.gamma = gamma

    def set_session(self, session):
        self.session = session
    
    def copy_from(self, other):
        ops = []
        my_params = self.params
        other_params = other.params
        for p, q in zip(my_params, other_params):
            actual =self.session.run(q)
            op = p.assign(actual)
            ops.append(op)
        self.session.run(ops)

        
    def predict(self, X):
        X = np.atleast_2d(X)
        return self.session.run(self.predict_op, feed_dict = {self.X:X})
    
    def train(self, target_network):
        if len(self.experience['s']<self.min_experience):
            return
    
        idx = np.random.choice(len(self.experience['s'], size = self.batch_sz, replace=False))
        states = [self.experience['s'][i] for i in idx]
        actions = [self.experience['a'][i] for i in idx]
        rewards = [self.experience['r'][i] for i in idx]
        next_states = [self.experience['s2'][i] for i in idx]
        next_Q = np.max(target_network.predict(next_states), axis =1)
        targets = [r+ self.gamma*next_q for r, next_q in zip(rewards, next_Q)]
        self.session.run(
            self.train_op, feed_dict={
                self.X: states, self.G: targets, self.actions: actions
            }
        )

    def add_experience(self, s, a, r, s2):
        if len(self.experience['s']) >= self.max_experiences:
            self.experience['s'].pop(0)
            self.experience['a'].pop(0)
            self.experience['r'].pop(0)
            self.experience['s2'].pop(0)
        self.experience['s'].append(s)
        self.experience['a'].append(a)
        self.experience['r'].append(r)
        self.experience['s2'].append(s2)
    
    def sample_action(self, x, eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            X = np.atleast_2d(x)
            return np.argmax(self.predict(X)[0])


def play_one(env, model, tmodel, eps, gamma, copy_period):
    observation, _ = env.reset()
    done = False
    totalreward = 0
    iters = 0
    while not done and iters<2000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, truncated, info = env.step(action)
        totalreward+=reward
        if done:
            reward-=200 
        model.add_experience(prev_observation, action, reward, observation)
        model.train(tmodel)
        iters+=1
        if iters%copy_period == 0:
            tmodel.copy_from(model)

def main():
    env = gym.make('CartPole-v0', render_mode = 'human')
    gamma = 0.99
    copy_period = 50

    D = len(env.observation_space.sample())
    K = env.action_space.n
    sizes = [200,200]
    model = DQN(D, K, sizes, gamma)
    tmodel = DQN(D, K, sizes, gamma)
    init = tf.global_variables_initializer()
    session = tf.InteractiveSession()
    session.run(init)
    model.set_session(session)
    tmodel.set_session(session)

main()