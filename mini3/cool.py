import numpy as np
from copy import copy
import matplotlib.pyplot as plt

'''
input_neurons : [(neuron, weight), ..., (neuron, weight)]
tau : between [0, 1], regulates inertia of neuron's input
default_value : value an input neuron outputs (either 0 or 1)
'''
class Neuron(object):
    def __init__(self, input_neurons = [], tau = 0.1, default_value = 1.0, N = 256):
        self.prevX = 0.0
        self.input_neurons = input_neurons
        self.tau = tau
        self.default_value = default_value
        self.Xs = {0 : 0}
        self.G = 1
        self.V = [0. for _ in range(N)]
        self.Vp = [0. for _ in range(N)]
    
    def activate(self, t):
        if len(self.input_neurons) > 0:
            return 1.0 / (1 + np.exp(-1.0 * self.G * self.X(t)))
        else:
            return self.default_value

    def X(self, t):
        if not t in self.Xs.keys():
            self.Xs[t] = self.Xs[t - 1] * (1 - self.tau) + self.tau * sum(map(lambda (neuron, weight) : weight * neuron.activate(t), self.input_neurons))
        return self.Xs[t]

    def add_input(self, neuron, weight):
        self.input_neurons.append((neuron, weight))
        
    def set_default(self, value):
        self.default_value = value

    def update_vg(self, t, rate = 0.5, NE = 0, alpha = 1):
        N = len(self.V)
        
        for i in range(len(self.V)):
            noise = np.random.normal(0, 1)
            noise = 0
            self.V[i] = rate * self.Vp[i] + (1 - rate) * (self.activate(t) + alpha * (sum(self.Vp) - N * self.Vp[i])+ noise) - NE
        self.G = sum(self.V)
        self.Vp = self.V
        self.V = [0. for _ in range(N)]
        
        

def main():
    input_neuron_t = Neuron(default_value = 0.0)
    input_neuron_d = Neuron(default_value = 1.0)
    
    decision_target = Neuron(input_neurons = [(input_neuron_t, 1)], tau = 0.5)
    decision_distract = Neuron(input_neurons = [(input_neuron_d, 1)], tau = 0.5)
    
    decision_target.add_input(decision_distract, -1)
    
    output_neuron = Neuron(input_neurons = [(decision_target, 1)], tau = 0.5)

    data = []
    D = 15
    T = 5
    
    for t in range(D):
        data.append(output_neuron.activate(t))
        decision_target.update_vg(t)
        print decision_target.G

    input_neuron_t.set_default(1.0)
    input_neuron_d.set_default(0.0)

    print "SWAP"

    for t in range(D, D + T):
        data.append(output_neuron.activate(t))
        decision_target.update_vg(t)
    data = np.array(data)

    plt.scatter(np.linspace(0, D + T, D + T), data)
    plt.show()
    
main()
