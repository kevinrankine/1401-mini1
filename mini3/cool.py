import numpy as np

'''
input_neurons : [(neuron, weight), ..., (neuron, weight)]
tau : between [0, 1], regulates inertia of neuron's input
default_value : value an input neuron outputs (either 0 or 1)
'''
class Neuron(object):
    def __init__(self, input_neurons = [], tau = 0.1, default_value = 1.0, lc_group = None):
        self.prevX = 0.0
        self.input_neurons = input_neurons
        self.tau = tau
        self.default_value = default_value
        self.Xs = {0 : 0}
        self.G = 1
    
    def activate(self, t):
        if len(self.input_neurons) > 0:
            return 1.0 / (1 + np.exp(self.G * self.X(t)))
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

def main():
    input_neuron_t = Neuron(default_value = 0.0)
    input_neuron_d = Neuron(default_value = 1.0)
    
    decision_target = Neuron(input_neurons = [(input_neuron_t, 1)], tau = 1)
    decision_distract = Neuron(input_neurons = [(input_neuron_d, 1)], tau = 1)
    
    decision_target.add_input(decision_distract, -1)
    print decision_target.activate(1)
    return 
    
    output_neuron = Neuron(input_neurons = [(decision_target, 1)])
    
    N = 256
    V = [0. for _ in range(N)]
    Vp = [0. for _ in range(N)]

    rate = 0.5
    NE = 10
    
    for t in range(100):
        print "Activation {}".format(output_neuron.activate(t))
        
        for i in range(N):
            V[i] = rate * Vp[i] - NE + decision_target.activate(t) + N * Vp[i] - sum(Vp) + np.random.randn()
        G = sum(V)
        
        print "G {}".format(G)
        decision_target.G = G
        Vp = V
        V = [0. for _ in range(N)]
    
main()
