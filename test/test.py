import neuralnet as nn
import numpy as  np

def test_autoencoder():
    shapes = [[8,3,8],[10,5,10],[30,10,30]]
    for nnshape in shapes:
        parD = nn.initialize(nnshape)
        alpha = 25
        x = np.identity(nnshape[0])    
        y = x
        #iterations parameter must be high enough for bigger shapes
        parD = nn.trainNN(parD,alpha,x,y,iterations=5000)
        autoout = nn.testNN(x,parD,outreturn=True)
        assert np.array_equal(np.around(autoout),x)