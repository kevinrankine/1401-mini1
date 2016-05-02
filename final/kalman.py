import numpy as np

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

    def update(self, x, reward):
        delta = reward - self.predict(x)
        self.deltas.append(delta)
        
        diffused_cov = self.cov + self.volatility * np.eye(self.d_in)
        k_gain = diffused_cov.dot(x) / (x.dot(diffused_cov.dot(x)) + self.var_r)
        cov_diff = k_gain[np.newaxis].T.dot(x.T.dot(diffused_cov)[np.newaxis])

        old_w = self.w
        self.w = self.w + k_gain * delta
        self.cov = diffused_cov - cov_diff
        
        self.var_r += self.rate / (2 * self.var_r) * ((delta ** 2 / self.var_r) - 1)
        self.volatility += self.rate / (2 * self.volatility) * ((np.linalg.norm(self.w - old_w) ** 2 / self.volatility) - self.d_in)

    def train(self, x, reward, nepochs=10):
        for i in range(nepochs):
            self.update(x, reward)
        
    def zero(self):
        self.w = np.zeros(self.d_in, dtype=np.float64)
        self.cov = self.var_w * np.eye(self.d_in, dtype=np.float64)
        
        self.var_w = self.old_var_w
        self.var_r = self.old_var_r
        self.volatility = self.old_volatility

    def predict(self, x):
        return np.dot(self.w, x)
        
