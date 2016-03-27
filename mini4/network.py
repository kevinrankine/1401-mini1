class Network(object):
    def __init__(self, hip, cort):
        self.hip = hip
        self.cort = cort

    def forward(self, inputs):
        self.hip.forward(inputs)
        self.cort.forward(inputs, self.hip.hiddens)

    def backward(self, outputs, learn = True):
        US = outputs[0]
        if learn:
            self.hip.backward(outputs)
        self.cort.backward(US)
