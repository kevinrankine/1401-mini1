import numpy as np

class Hippo(object):
    def __init__(self, d_in, d_hid):
        self.W1 = (np.random.rand(d_hid, d_in) - 0.5) / (0.3/0.5)
        self.dW1 = np.zeros((d_hid, d_in))
        
        self.W2 = (np.random.rand(d_in, d_hid) - 0.5) / (0.3/0.5)
        self.dW2 = np.zeros((d_in, d_hid))
        
        self.hiddens = np.zeros(d_hid)
        self.outputs = np.zeros(d_in)
        
        self.d_in = d_in
        self.d_hid = d_hid
        self.inputs = None

    def forward(self, inputs):
        
        self.hiddens = logistic(np.dot(self.W1, inputs))
        self.outputs = logistic(np.dot(self.W2, self.hiddens))
        self.inputs = inputs

    def backward(self, outputs):
        delta_out = (outputs - self.outputs) * (1 - self.outputs)
        delta_hid = self.hiddens * (1 - self.hiddens) * np.dot(delta_out,
                                                               self.W2)
        beta = 0.05 if outputs[0] == 1 else 0.005

        for i in xrange(self.d_in):
            for j in xrange(self.d_hid):
                self.dW2[i][j] = beta * delta_out[i] * self.hiddens[j] +\
                                 0.9 * self.dW2[i][j]
                self.W2[i][j] += self.dW2[i][j]

                self.dW1[j][i] = beta * delta_hid[j] * self.inputs[i] +\
                                 0.9 * self.dW1[j][i]
                self.W1[j][i] += self.dW1[j][i]

def logistic(x):
    return 1.0 / (1 + np.exp(-x))
