import gym
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_approximation import RBFSampler
import random
import numpy as np
import tensorflow as tf

class SGDRegressor:

    def __init__(self, D) -> None:
        lr = 10e-2
        self.w = tf.Variable(tf.random_normal(shape=(D, 1)), name='w')
        self.X = tf.placeholder(tf.float32, shape=(None, D), name='X')
        self.Y = tf.placeholder(tf.float32, shape =(None,), name='Y')
        
        Y_hat = tf.reshape(tf.matmul(self.X, self.w), [-1])
        delta = self.Y-Y_hat
        cost = tf.reduce_sum(delta**2)

        self.train_op = tf.train.GradientDescentOptimizer(lr).minimize(cost)
        self.predict_op = Y_hat

        init = tf.global_variables_initializer()
        self.session = tf.InteractiveSession()
        self.session.run(init)


    def partial_fit(self, X, Y):
        self.session.run(self.train_op, feed_dict={self.X:X, self.Y:Y})
        #self.w += self.lr*(Y-X.dot(self.w)).dot(X)

    def predict(self, X):

        return self.session.run(self.predict_op, feed_dict={self.X:X})
class FeatureTranformer:
    def __init__(self, env):
        examples = np.random.random((20000, 4))*2-2
        scaler = StandardScaler()
        featurizer = FeatureUnion([
            ("rbf1", RBFSampler(gamma=0.05, n_components=1000)),
            ("rbf2", RBFSampler(gamma=1.0, n_components=1000)),
            ("rbf3", RBFSampler(gamma=0.5, n_components=1000)),
            ("rbf4", RBFSampler(gamma=0.1, n_components=1000)),
        ])
        feature_examples = featurizer.fit_transform(scaler.fit_transform(examples))
        self.dimensions = feature_examples.shape[1]
        self.featurizer =featurizer
        self.scaler = scaler

    def transform(self, observations):
        return self.featurizer.transform(self.scaler.transform(observations))


class Model:
    def __init__(self, env, ft):
        models = []

        for _ in range(env.action_space.n):
            model = SGDRegressor(ft.dimensions) 
            #model.partial_fit(ft.transform([env.reset()[0]]), [0])
            models.append(model)
        self.models = models
        self.featureTransformer= ft
        self.env = env
        
    def predict(self, observation):
        observation = self.featureTransformer.transform(observation.reshape(1,-1))
        return np.array([m.predict(observation)[0] for m in self.models])

    def update(self, observation, action, G):
        observation = self.featureTransformer.transform(observation.reshape(1,-1))
        self.models[action].partial_fit(observation, [G])

    def sample_action(self, observation, eps):
        if random.random()<eps:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.predict(observation))
   
def runOnce(env, model, eps, gamma):
    totalrewards= 0
    observation, _ = env.reset()
    N=2000
    step=0
    terminated=False
    while not terminated and step<N:
        action = model.sample_action(observation, eps)  # User-defined policy function
        prev_observation = observation
        observation, reward, terminated, truncated, info = env.step(action)
        if terminated:
            reward=-200
        totalrewards+=reward
        G = reward+gamma*np.max(model.predict(observation))
        model.update(prev_observation, action, G)
        step+=1
    return totalrewards

def main():
   env = gym.make('CartPole-v0', render_mode='human')
   gamma = 0.9
   ft = FeatureTranformer(env)
   model = Model(env, ft)
   episodes = 500
   rewards = [0 for _ in range(episodes)]
   for i in range(episodes):
       eps = 1.0/((i+1)**0.5)
       rewards[i] = runOnce(env, model, eps, gamma)
       print(rewards[i])
   print(rewards)
   env.close()

if __name__ =="__main__":
    main()

