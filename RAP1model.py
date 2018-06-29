import seqio
from neuralnet import sigmoid, sigderiv, initialize, \
                      forback, trainNN, testNN
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn import metrics
import sys
from pprint import pprint

files = ['data/rap1-lieb-positives.txt','data/yeast-upstream-1k-negative.fa']
allfileseqs = seqio.getseqs(files) # read in all the seqeunces
positives, negatives = allfileseqs[0], allfileseqs[1]

newnegs = list() #  get rid of negatives sequences that are also postiive sequences
for seq in negatives:
    if seq not in positives:
        newnegs.append(seq)
negatives = newnegs
negatives = list(np.random.choice(negatives, 137))


allseqs = np.append(positives,negatives) # make them all into one long sequence
binarr = seqio.onehot(allseqs) # convert to binary


positiveys = np.ones([len(positives)]) # make the "y" data
negativeys = np.zeros([len(negatives)])
allys = np.append(positiveys,negativeys)


randindex = list(range(0,len(allseqs))) #  shuffle the data so in no particular order
np.random.shuffle(randindex)
inputarray = binarr[randindex,:]
outputarray = allys[randindex]

input_layer_size = 68 # make the architecture match the desired input and output
hidden_layer_size = 34
output_layer_size = 1
nnshape=[input_layer_size,hidden_layer_size,output_layer_size]

alpha = 1
kfoldsplit = 2

skf = KFold(n_splits=kfoldsplit)
count = 1   
AUCs = list()
print('\n########################## Learning started ##########################')
for alpha in [1] + list(range(5,100,5)):
    for train, test in skf.split(inputarray, outputarray):
        print('(train, test) set %d of %d K-fold cross-validation sets:' % (count,kfoldsplit))

        intrain = inputarray[train] 
        outtrain = outputarray[train]

        intest = inputarray[test]
        outtest = outputarray[test]

        parD = initialize(nnshape)

        batches = list(range(10,len(outtrain),10)) # train in batches of 10 
        for iterations in tqdm(range(3000)):
            for index, batch in enumerate(batches):
                if index == 0:
                    batch_intrain = intrain[:batch,:]
                    batch_outtrain = np.matrix(outtrain[:batch]).T
                elif batch == batches[-1]:
                    batch_intrain = intrain[batch:,:]
                    batch_outtrain = np.matrix(outtrain[batch:]).T
                else:
                    batch_intrain = intrain[batches[index-1]:batches[index],:]
                    batch_outtrain = np.matrix(outtrain[batches[index-1]:batches[index]]).T

                parD = forback(parD, alpha,
                               batch_intrain, 
                               batch_outtrain, 
                               nnshape)

        AUC = testNN(intest,parD,AUC=True,y=outtest)
        AUCs.append([alpha, AUC])
        count += 1
np.save("AUCs",np.array(AUCs))
allseqs = seqio.getseqs("rap1-lieb-test.txt")[0]
binarr = seqio.onehot(allseqs)

with open("predictions.txt","w") as f:
    for i, seq in enumerate(allseqs):
        output = float(testNN(binarr[i],parD,outreturn=True))
        f.write(seq + "\t" + "{:5.4f}".format(output) + "\n")