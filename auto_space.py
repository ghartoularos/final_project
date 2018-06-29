import neuralnet as nn
import numpy as  np

shapes = [[8,3,8],[10,5,10],[30,10,30],[8,40,8]]
shape  = -1
for nnshape in shapes:
    shape += 1
    datacollect = list()
    for alpha in [1] + list(range(5,101,5)):
        for iterations in range(100,5001,100):
            parD = nn.initialize(nnshape)
            x = np.identity(nnshape[0])    
            y = x
            #iterations parameter must be high enough for bigger shapes
            parD = nn.trainNN(parD,alpha,x,y,iterations=iterations)
            autoout = nn.testNN(x,parD,outreturn=True)
            avgerr = np.average(abs(autoout - x))
            datacollect.append([shape,alpha,iterations,avgerr])
    np.save("%d_data" % shape,np.array(datacollect))

