import gym
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from gym import wrappers
from datetime import datetime
from q_learning_mountain_car import plot_cost_to_go, plot_running_avg, FeatureTransformer

class BaseModel:
    def __init__(self, D):
        self.w = np.random.randn(D) / np.sqrt(D)

    def partial_fit(self, input_, target, eligibility, lr=10e-3):
        self.w += lr*(target-input_.dot(self.w))*eligibility
    
    def predict(self, X):
        X = np.array(X)
        return X.dot(self.w)
    
class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        D = feature_transformer.dimensions
        self.eligibilities = np.zeros((env.action_space.n, D))
        for i in range(env.action_space.n):
            model = BaseModel(D)
            self.models.append(model)


    def predict(self, input):
        X = self.feature_transformer.transform([input])
        return np.array([model.predict(X)[0] for model in self.models])
         
    def update(self, s, a, G, gamma, lambda_):
        X = self.feature_transformer.transform([s])
       
        self.eligibilities *= gamma*lambda_
        self.eligibilities[a] += X[0]
        self.models[a].partial_fit(X[0], G, self.eligibilities[a])
    
    def sample_action(self, s, eps):
        if np.random.random()>eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(s))
        
def play_one(model, eps, gamma, lambda_):

    observation = env.reset()[0]
    done = False
    totalreward = 0
    iters = 0

    while not done and iters<10000:
        action = model.sample_action(observation, eps)
        prev_observation = observation
        observation, reward, done, truncated, info = env.step(action)

        G = reward+gamma*np.max(model.predict(observation)[0])
        model.update(prev_observation, action, G, gamma, lambda_)

        totalreward+=reward
        iters+=1

if __name__ == "__main__":

    env = gym.make('MountainCar-v0', render_mode = 'human')
    ft = FeatureTransformer(env)
    model = Model(env, ft)
    gamma = 0.99
    lambda_ = 0.7
    N=300
    totalrewards= np.empty(N)
    for n in range(N):
        eps = 0.1*(0.97**n)
        totalreward = play_one(model, eps, gamma, lambda_)
        totalrewards[n] = totalreward

