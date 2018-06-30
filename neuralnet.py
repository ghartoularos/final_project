import numpy as np
import sys
from tqdm import tqdm
from pprint import pprint
from sklearn import metrics
import warnings

def sigmoid(x):  # sigmoid function
    return(1/(1+np.exp(-x)))

def sigderiv(x): # the derivative of the sigmoid is simply its evaluation multiplied by its complement
    return x*(1 - x)

def initialize(nnshape): # function for intializing the weights and bias matrices
    parD = {} # initiailize the parameter dictionary that will hold all model parameters
    for layer in range(len(nnshape)-1): # intialize weight vectors
        parD[(layer,'W')] = np.zeros([nnshape[layer],nnshape[layer +  1]])
        W = parD[(layer,'W')]

        for i in range(len(W)): # fill in weight, bias vectors with normally distributed noise
            W[i,:] = np.random.normal(0,0.01,nnshape[layer+1])

        parD[(layer,'B')] = np.random.normal(0,0.01,nnshape[layer + 1])
    parD['nnshape'] = nnshape # make one more k to just store shape so it doesn't need to be passed separately
    return parD

def forback(parD,alpha,x,y,nnshape): # the forward pass, back propogation steps
    '''
    This can be called multiple times to learn the best parameter dictionary
    '''
    parD[(0,'a')] = x # intialize the "activation" from layer 0 which is just the input
    # This next for loop is the forward pass through the model
    for layer in range(len(nnshape)-1):
        parD[(layer,'dw')] = np.zeros([nnshape[layer],nnshape[layer + 1]])
        parD[(layer,'db')]= np.zeros([1,nnshape[layer + 1]])
        # output of the current layer is the dot product of the activaiton of the last layer times the weight + the bias
        parD[(layer + 1,'o')] = np.dot(parD[(layer,'a')],parD[(layer,'W')])+parD[(layer,'B')]
        # activation of the current layer is the sigmoid of the output of the current layer
        parD[(layer + 1,'a')] = sigmoid(parD[(layer + 1,'o')])

    # This next for loop is the back propogation
    for layer in range(len(nnshape)-1,0,-1): # going backwards
        parD[(layer,'e')] = sigderiv(parD[(layer,'a')]) # what's the error and in what "direciton" is it moving?
        if layer == len(nnshape) - 1: # for the last layer...
            parD[(layer,'e')] = np.multiply((parD[(2,'a')] - y),parD[(layer,'e')])
        else: # for all intermediate and first layers...
            parD[(layer,'e')] = np.multiply(np.dot(parD[(layer + 1,'e')],parD[(layer,'W')].T),parD[(layer,'e')])

        for i in range(parD[(0,'a')].shape[0]): # add the delta w and delta b 
            parD[(layer - 1,'dw')] += np.multiply(np.matrix(parD[(layer - 1,'a')][i]).T, np.matrix(parD[(layer,'e')][i])) 
            parD[(layer - 1,'db')] += parD[(layer,'e')][i] 

    for layer in range(len(nnshape) - 1): # update the weights, biases with the modifications
        parD[(layer,'W')] = parD[(layer,'W')] - alpha*(1/nnshape[0] * parD[(layer,'dw')])
        parD[(layer,'B')] = parD[(layer,'B')] - alpha*(1/nnshape[0] * parD[(layer,'db')])
    return parD

def trainNN(parD,alpha,x,y,iterations=1000): # train the NN for as many iterations as you'd like
    nnshape = parD['nnshape']
    for _ in tqdm(range(iterations)):
        parD = forback(parD,alpha,x,y,nnshape)
    return parD

def testNN(x,parD,AUC=False,y='null',outreturn=False):
    '''
    If AUC is set to True, only AUC will be printed.
    If outreturn == True, the output will be returned.
    If outreturn == False, the outeput will be printed.
    '''
    with warnings.catch_warnings(): # this error when comparing an np.array to a string
        warnings.simplefilter(action='ignore', category=FutureWarning)
        if AUC == True and y == 'null':
            pprint('Supply "y" values for AUC.')
            raise SystemExit
    nnshape = parD['nnshape']
    parD[(0,'a')] = x
    for layer in range(len(nnshape)-1): # following all steps as in training, forward pass
        parD[(layer + 1,'o')] = np.dot(parD[(layer,'a')],parD[(layer,'W')])+parD[(layer,'B')]
        parD[(layer + 1,'a')] = sigmoid(parD[(layer + 1,'o')])
    if AUC == True: 
        AUC = metrics.roc_auc_score(y, parD[(layer + 1,'a')])
        pprint('AUC: {:5.4f}'.format(AUC))
        return AUC
    else:
        if outreturn == True:
            return parD[(layer + 1,'a')]
        else:
            print("\nInput:")
            print(x)
            print("\nOutput:")
            print(parD[(layer + 1,'a')])
    return

if __name__ == '__main__':
    if sys.argv[1] == 'auto':
        nnshape = [10,5,10] # this is arbitrary, could be 8x3x8 or anything else really
        parD = initialize(nnshape)
        alpha = 50 # went with 50 here, but it didn't matter between 1 and 100
        x = np.identity(nnshape[0])    
        y = x
        parD = trainNN(parD,alpha,x,y,iterations=int(sys.argv[2]))
        testNN(x,parD)
    else:
        None

