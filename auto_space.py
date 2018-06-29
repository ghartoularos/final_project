import neuralnet as nn
import numpy as  np

shapes = [[8,3,8],[10,5,10],[30,10,30],[100,40,100]]
for nnshape in shapes:
    for alpha in [1] + list(range(5,101,5)):
        for iterations in range(100,5001,100):
            parD = nn.initialize(nnshape)
            alpha = 25
            x = np.identity(nnshape[0])    
            y = x
            #iterations parameter must be high enough for bigger shapes
            parD = nn.trainNN(parD,alpha,x,y,iterations=iterations)
            autoout = nn.testNN(x,parD,outreturn=True)
            assert np.array_equal(np.around(autoout),x)