import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use('ggplot')
mpl.rcParams['figure.figsize'] = (8., 6.)


class KalmanFilter(object):
    def __init__(self, d_in, var_w=1, var_r=1, volatility=0.01, rate = 0.0):
        self.d_in = d_in
        self.var_w = var_w
        self.var_r = var_r
        self.volatility = volatility
        self.old_var_w = var_w
        self.old_var_r = var_r
        self.old_volatility = volatility
        self.rate = rate

        self.w = np.zeros(d_in, dtype=np.float64)
        self.cov = var_w * np.eye(d_in, dtype=np.float64)
        self.deltas = []
        self.kalmans = []

    def update(self, x, reward):
        delta = reward - self.predict(x)
        
        diffused_cov = self.cov + self.volatility * np.eye(self.d_in)
        k_gain = diffused_cov.dot(x) / (x.dot(diffused_cov.dot(x)) + self.var_r)
        cov_diff = k_gain[np.newaxis].T.dot(x.T.dot(diffused_cov)[np.newaxis])

        old_w = self.w
        self.w = self.w + k_gain * delta
        self.cov = diffused_cov - cov_diff
        
        self.var_r += self.rate / (2 * self.var_r) * ((delta ** 2 / self.var_r) - 1)
        self.volatility += self.rate / (2 * self.volatility) * ((np.linalg.norm(self.w - old_w) ** 2 / self.volatility) - self.d_in)

        self.var_r = max(self.var_r, 0.01)
        self.volatility = max(self.volatility, 0.01)

        self.deltas.append(delta)
        self.kalmans.append(map(lambda x : x.tolist(), k_gain))

    def train(self, x, reward, nepochs=10):
        for i in range(nepochs):
            self.update(x, reward)
        
    def zero(self):
        self.w = np.zeros(self.d_in, dtype=np.float64)
        self.cov = self.var_w * np.eye(self.d_in, dtype=np.float64)
        
        self.var_w = self.old_var_w
        self.var_r = self.old_var_r
        self.volatility = self.old_volatility
        self.deltas = []
        self.kalmans = []

    def predict(self, x):
        return np.dot(self.w, x)

    def plot(self):
        plt.plot(np.arange(1, len(self.deltas) + 1), self.deltas, label='$\delta_n$')
        plt.plot(np.arange(1, len(self.kalmans) + 1), zip(*self.kalmans)[0], label='$k_A$')
        if len(self.kalmans[0]) > 1:
            plt.plot(np.arange(1, len(self.kalmans) + 1), zip(*self.kalmans)[1], label='$k_B$')
        
        plt.legend()
        plt.ylim(-3, 3)
        plt.show()
