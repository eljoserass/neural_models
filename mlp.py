import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.metrics import mean_squared_error


data = load_diabetes()

x = np.array(data.data)[:5] # 10 features
y = np.array(data.target)[:5] # 1 label


# 2 layer mlp, with 1 perceptron each layer
mlp_config = [
    (x.shape[1], 1),
    (1, 1)
]


def init_layers(mlp_config: list) -> list:
    init_num = 1

    mlp = [(
            np.full((layer[0], layer[1]), init_num), # init weights
            np.full((layer[1]), init_num) # init biases
            ) for layer in mlp_config]
    
    return mlp


class MLP:
    def __init__(self, mlp_config:list, input_shape:int = None, output_shape:int = None): # after add some config for the layers in between

        self.layers = init_layers(mlp_config=mlp_config)
        self.z = [] # logits
        self.a = [] # activations
        self.errors = []
        self.act = lambda x: x # identity activation
        self.dact = lambda x: np.ones_like(x)
        self.dcost = lambda pred, y: pred - y
    
    def forward(self, x):
        feed = x.copy()
        self.z = []
        for l in self.layers:
            z = feed@l[0]+l[1] # x@w+b
            self.z.append(z)
            
            feed = self.act(z)
            self.a.append(feed) 
        return feed

    def backward(self, y):
        
        # probably in verse order
        L_error = self.dcost(self.a[-1], y) * self.dact(self.z[-1])
        
        l_count = len(self.layers) - 1
        
        self.errors.append(L_error)
        
        while l_count > -1:
            index = l_count - 1
            L_error = (self.layers[index + 1][0].T * L_error) * self.dact(self.z[index])
            self.errors.append(L_error)
            
            l_count = -1
        self.errors.reverse()

mlp = MLP(mlp_config=mlp_config)
mlp.forward(x)
mlp.backward(y)
print (mlp.errors)

