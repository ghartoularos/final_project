import numpy as np
from Bio import SeqIO

def getseqs(filenames):

    if type(filenames) == str:
        filenames = [filenames]

    allfileseqs = list()
    for filename in filenames:

        if filename.split('.')[-1] == 'fa':
            fasta = True
        else:
            fasta = False

        with open(filename,'r') as f:

            if fasta == False:
                allseqs = [i.strip() for i in f.readlines()]
            else:
                allseqs = list()
                for sequence in SeqIO.parse(filename,"fasta"):
                    seq = str(sequence.seq)
                    if len(seq) < 17:
                        continue
                    else:
                        startsplice = np.random.randint(0,len(seq)-16)
                        seq = seq[startsplice:startsplice + 17]
                    allseqs.append(seq)

        allfileseqs.append(allseqs)
    return(allfileseqs)

def onehot(seqs):

    # use one hot encoding
    alphabet = {'A': (1, 0, 0, 0), 
                'C': (0, 1, 0, 0),
                'G': (0, 0, 1, 0), 
                'T': (0, 0, 0, 1)}
    binarr = np.array([[bit for char in seq
                        for bit in alphabet[char]]
                        for seq in seqs], dtype=np.int)
    return(binarr)