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
    alphabet = {'A':'1000',
                'C':'0100',
                'G':'0010',
                'T':'0001'}

    newseqs = list()
    for seq in seqs:
        newseq = list()
        for NT in seq:
            newseq.append(alphabet[NT])
        newseqs.append(newseq)
    seqs = newseqs

    binarr = np.zeros([len(seqs),len(seqs[0])*4])
    for i in range(len(binarr)):
        binarr[i,:] = list(''.join(seqs[i]))
    return(binarr)