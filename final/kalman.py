import numpy as np

class KalmanFilter(object):
    def __init__(self, d_in, var_w=1, var_r=1, volatility=0.01):
        self.d_in = d_in
        self.var_w = var_w
        self.var_r = var_r
        self.volatility = volatility

        self.w = np.zeros(d_in, dtype=np.float64)
        self.cov = var_w * np.eye(d_in, dtype=np.float64)

    def update(self, x, reward):
        delta = reward - self.predict(x)
        diffused_cov = self.cov + self.volatility * np.eye(self.d_in)
        k_gain = diffused_cov.dot(x) / (x.dot(diffused_cov.dot(x)) + self.var_r)
        cov_diff = k_gain[np.newaxis].T.dot(x.T.dot(diffused_cov)[np.newaxis])
        
        self.w = self.w + k_gain * delta
        self.cov = diffused_cov - cov_diff

    def train(self, x, reward, nepochs=10):
        for i in range(nepochs):
            self.update(x, reward)
        
    def zero(self):
        self.w = np.zeros(self.d_in, dtype=np.float64)
        self.cov = self.var_w * np.eye(self.d_in, dtype=np.float64)

    def predict(self, x):
        return np.dot(self.w, x)
        
