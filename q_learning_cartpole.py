import gym
from gym import wrappers
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

#[1,2,3] -> 123
def build_state(features):
    return int("".join(map(lambda x: str(x), features)))

#figures out which value the bin belongs in
def to_bin(value, bins):
    return np.digitize(x=[value], bins = bins)[0]

class FeatureTransformer:
    def __init__(self):
        self.cart_position_bins = np.linspace(-2.4,2.4,9)
        self.cart_velocity_bins = np.linspace(-2, 2, 9)
        self.pole_angle_bins = np.linspace(-0.4, 0.4, 9)
        self.pole_velocity = np.linspace(-3.5,3.5,9)
    
    def transform(self, observation):
        cart_pos, cart_vel, pole_angle, pole_vel= observation
        return build_state([
            to_bin(cart_pos, self.cart_position_bins),
            to_bin(cart_vel, self.cart_velocity_bins), 
            to_bin(pole_angle, self.pole_angle_bins),
            to_bin(pole_vel, self.pole_velocity)
        ])

class Model:
    def __init__(self, env, feature_transformer):
        self.env = env
        self.feature_transformer = feature_transformer
        num_states = 10**env.observation_space.shape[0]
        
        num_actions = env.action_space.n
        self.Q = np.random.uniform(low=-1, high =1, size = (num_states, num_actions))

    def predict(self, s):
        x = self.feature_transformer.transform(s)
        #returns a one-d array of all actions
        return self.Q[x]
    
    def update(self, s, a, G):
        x= self.feature_transformer.transform(s)
        #update using Gradient Descent
        self.Q[x,a] += 10e-3*(G-self.Q[x, a])

    def sample_action(self, s, epsilon):
        if np.random.random()< epsilon:
            return self.env.action_space.sample()
        else:
            actions = self.predict(s)
            return np.argmax(actions)


def play_one(model, eps, gamma):

    observation = env.reset()[0]
    done = False
    totalreward = 0
    iters = 0
    while not done and iters<10000:
        #take action
        action  = model.sample_action(observation, eps)
        prev_observation = observation
        #observe environment after action taken
        observation, reward, done, _, info = env.step(action)
        totalreward+=reward
        if done and iters<199:
            reward = -300
        #update agent based on rewards from observation
        G = reward+gamma*np.max(model.predict(observation))
        model.update(prev_observation, action, G)
        iters+=1

    return totalreward


def plot_running_avg(totalrewards):
    N=len(totalrewards)
    running_avg = np.empty(N)
    for t in range(N):
        running_avg[t] = totalrewards[max(0,t-100):(t+1)].mean()
    plt.plot(running_avg)
    plt.title("Running Average")
    plt.show()

if __name__ == "__main__":
    
    if 'render' not in sys.argv:
        env=gym.make('CartPole-v0')
    else:
        env=gym.make('CartPole-v0', render_mode="human")
    if 'record' in sys.argv:
        env = wrappers.RecordVideo(env, 'my_awesome_dir')

    ft = FeatureTransformer()
    model = Model(env, ft)
    gamma = 0.9
    N=10000
    totalrewards = np.empty(N)
    for i in range(N):
        #chooses less randomly over episodes
        eps = 1.0/(i+1)**(0.5)
        totalrewards[i] = play_one(model, eps, gamma)
        if i%100 == 0:
            print("episode", i, "has reward", totalrewards[i])
    # print("avg rewards for the last 100 episodes:", totalrewards[-100:].mean())
    # print("total steps:", totalrewards[-100:].mean())
    env.close()
    plot_running_avg(totalrewards)
    
