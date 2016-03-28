import numpy as np

class Cortex(object):
    def __init__(self, d_in, d_hid, d_out, d_feats):
        self.W1 = (np.random.rand(d_hid, d_in) - 0.5) / (0.3/0.5)
        self.W2 = (np.random.rand(d_out, d_hid) - 0.5) / (0.3/0.5)
        self.W_feats = (np.random.rand(d_hid, d_feats) - 0.5) / (0.3/0.5)
        
        self.hiddens = np.zeros(d_hid)
        self.outputs = np.zeros(d_out)
        
        self.d_in = d_in
        self.d_hid = d_hid
        self.d_out = d_out
        self.d_feats = d_feats
        self.inputs = None

    def forward(self, inputs, features):
        self.hiddens = logistic(np.dot(self.W1, inputs), y=np.dot(self.W_feats, features))
        self.outputs = logistic(np.dot(self.W2, self.hiddens))
        self.features = features
        self.inputs = inputs

    def backward(self, US):
        beta = 0.5 if US == 1 else 0.05

        for i in range(self.d_out):
            for j in range(self.d_hid):
                self.W2[i][j] += beta * (US - self.outputs[i])*(self.hiddens[j])

        delta = np.dot(self.W_feats, self.features)
        for i in range(self.d_hid):
            for j in range(self.d_in):
                self.W1[i][j] += beta * (delta[i] - self.hiddens[i]) *\
                                self.inputs[j]
        
def logistic(x, y = None):
    if y is None:
        return 1.0 / (1 + np.exp(-x))
    else:
        return 1.0 / (1 + np.exp(-x)*np.exp(-y))
