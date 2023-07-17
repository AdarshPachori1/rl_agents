import gym
import os, sys
import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor
import matplotlib.pyplot as plt
from datetime import datetime
from gym import wrappers
import random

class SGDRegressor:
    def __init__(self, learning_rate):

        self.w = None
        if learning_rate == 'constant':
            self.lr = 10e-3
   
    def partial_fit(self, X, Y):
        if self.w is None:
           D= X.shape[1]
           self.w = np.random.randn(D)/np.sqrt(D)
        self.w+= self.lr*(Y-X.dot(self.w)).dot(X)

    def predict(self,X):
        return X.dot(self.w)
   


class FeatureTransformer:
    def __init__(self, env):
        observation_examples=np.array([env.observation_space.sample() for x in range(10000)])
        if verbose:
            print("observation examples.shape:", observation_examples.shape)
        scaler = StandardScaler()
        scaler.fit(observation_examples)

        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma = 5.0, n_components = 500)),
            ("rbf2", RBFSampler(gamma = 2.0, n_components = 500)),
            ("rbf3", RBFSampler(gamma = 1.0, n_components = 500)),
            ("rbf4", RBFSampler(gamma = 0.5, n_components = 500))
        ])
        scaled = scaler.transform(observation_examples)
        if verbose:
            print("scaled.shape:", scaled.shape)
        example_features = featurizer.fit_transform(scaled)
        self.scaler = scaler
        self.featurizer = featurizer

    def transform(self, observations):
        return self.featurizer.transform(self.scaler.transform(observations))
    

class Model:
    def __init__(self, env, feature_transformer, learning_rate):
        self.env = env
        self.models = []
        self.feature_transformer = feature_transformer
        for i in range(env.action_space.n):
            model = SGDRegressor(learning_rate=learning_rate)
            model.partial_fit(feature_transformer.transform([env.reset()[0]]), [0])
            self.models.append(model)

    def predict(self, s):
        if verbose:
            print("[s].shape:", (1,)+s.shape)
        X = self.feature_transformer.transform([s])
        assert(len(X.shape)==2)
        return np.array([m.predict(X)[0] for m in self.models])
    
    def update(self, s, a, G):
        X = self.feature_transformer.transform([s])
        if verbose:
            print("[s].shape:", (1,)+s.shape)
            print("x.shape:", X.shape)
        assert(len(X.shape)==2)
        self.models[a].partial_fit(X, [G])
    
    def sample_action(self, s, eps):
        if np.random.random() < eps:
            return self.env.action_space.sample()
        else:

            accelerating_dir = np.argmax(self.predict(s))
            if verbose and random.random()<0.1:
                if accelerating_dir==0:
                    print("left")
                elif accelerating_dir==1:
                    print("nothing")
                else:
                    print("right")
            return accelerating_dir

def play_one(model, env, eps, gamma, n=5):
    observation = env.reset()[0]
    done =0
    totalreward=0
    rewards=[]
    states=[]
    actions=[]
    iters=0
    
    multiplier = np.array([gamma]*n)**np.arange(n)
    
    #changing things
    while not done and iters < 10000:
        action = model.sample_action(observation, eps)

        states.append(observation)
        actions.append(action)
        prev_observation = observation 
        observation, reward, done , truncated,  info = env.step(action)

        rewards.append(reward)

        if len(rewards)>=n:
            return_up_to_prediction = multiplier.dot(rewards[-n:])
            G = return_up_to_prediction+gamma**n*np.max(model.predict(observation)[0])
            model.update(states[-n], actions[-n], G)
        totalreward+=reward
        iters+=1
    return totalreward

def plot_cost_to_go(env, estimator, num_tiles=20):
  x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
  y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
  X, Y = np.meshgrid(x, y)
  # both X and Y will be of shape (num_tiles, num_tiles)
  Z = np.apply_along_axis(lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))
  # Z will also be of shape (num_tiles, num_tiles)

  fig = plt.figure(figsize=(10, 5))
  ax = fig.add_subplot(111, projection='3d')
  surf = ax.plot_surface(X, Y, Z,
    rstride=1, cstride=1, cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
  ax.set_xlabel('Position')
  ax.set_ylabel('Velocity')
  ax.set_zlabel('Cost-To-Go == -V(s)')
  ax.set_title("Cost-To-Go Function")
  fig.colorbar(surf)
  plt.show()


def plot_running_avg(totalrewards):
  N = len(totalrewards)
  running_avg = np.empty(N)
  for t in range(N):
    running_avg[t] = totalrewards[max(0, t-100):(t+1)].mean()
  plt.plot(running_avg)
  plt.title("Running Average")
  plt.show()



verbose = False
def main(show_plots=True):
  global verbose
  if 'render' not in sys.argv:
    env=gym.make('MountainCar-v0')
  else:
    env=gym.make('MountainCar-v0', render_mode="human")
  if 'record' in sys.argv:
    filename = os.path.basename(__file__).split('.')[0]
    monitor_dir = './' + filename + '_' + str(datetime.now())
    env = wrappers.Monitor(env, monitor_dir)

  if 'verbose' in sys.argv:
     verbose =True
  ft = FeatureTransformer(env)
  model = Model(env, ft, "constant")
  gamma = 0.99

  N = 10
  totalrewards = np.empty(N)
  for n in range(N):
    # eps = 1.0/(0.1*n+1)
    eps = 0.1*(0.97**n)
    if n == 199:
      print("eps:", eps)
    # eps = 1.0/np.sqrt(n+1)
    totalreward = play_one(model, env, eps, gamma)
    totalrewards[n] = totalreward
    if (n + 1) % 100 == 0:
      print("episode:", n, "total reward:", totalreward)
  #print("avg reward for last 100 episodes:", totalrewards[-100:].mean())
  #print("total steps:", -totalrewards.sum())
  env.close()
  if show_plots:
    plt.plot(totalrewards)
    plt.title("Rewards")
    plt.show()

    plot_running_avg(totalrewards)

    # plot the optimal state-value function
    plot_cost_to_go(env, model)


if __name__ == '__main__':
  main()