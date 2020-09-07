import pandas as pd
import numpy as np
import random
import zipfile
import os

def generate_chessboard(size1, size2, frac=0.25, seed=10):
    #generates a data set with one uniformly drawn feature
    #from range 0...100, and y based on a xor function on the
    #parity of x. frac(tion) of pairs from all possible pairwise
    #combinations are generated
    np.random.seed(seed)
    random.seed(seed)
    X1 = 100*np.random.random(size1)
    X2 = 100*np.random.random(size2)
    paircount = int(frac*size1*size2)
    rowind = np.random.randint(0, size1, paircount)
    colind = np.random.randint(0, size2, paircount)
    #foo = rowind + colind
    #I = np.argsort(foo)
    #I = np.argsort(colind)
    I = np.argsort(colind)
    rowind = rowind[I]
    colind = colind[I]
    Y = []
    for rind, cind in zip(rowind, colind):
        y1 = int(X1[rind])%2
        y2 = int(X2[cind])%2
        y = np.logical_xor(y1, y2)
        if random.random() > 0.2:   
            Y.append(y)
        else:
            Y.append(y == False)
    Y = 2.*np.array(Y)-1
    X1 = X1.reshape(size1, 1)
    X2 = X2.reshape(size2, 1)
    rowind = np.array(rowind, dtype=np.int32)
    colind = np.array(colind, dtype=np.int32)
    return X1, X2, rowind, colind, Y

def generate_chessboard_singdom(size, frac=0.25, seed=10):
    #generates a data set with one uniformly drawn feature
    #from range 0...100, and y based on a xor function on the
    #parity of x. frac(tion) of pairs from all possible pairwise
    #combinations are generated
    np.random.seed(seed)
    random.seed(seed)
    X = 100*np.random.random(size)
    paircount = int(frac*size**2)
    rowind = np.random.randint(0, size, paircount)
    colind = np.random.randint(0, size, paircount)
    I = np.argsort(colind)
    rowind = rowind[I]
    colind = colind[I]
    Y = []
    for rind, cind in zip(rowind, colind):
        y1 = int(X[rind])%2
        y2 = int(X[cind])%2
        y = np.logical_xor(y1, y2)
        if random.random() > 0.2:   
            Y.append(y)
        else:
            Y.append(y == False)
    Y = 2.*np.array(Y)-1
    X = X.reshape(size, 1)
    rowind = np.array(rowind, dtype=np.int32)
    colind = np.array(colind, dtype=np.int32)
    return X, rowind, colind, Y

def load_metz(binarize=True, pct=None):
    Y = np.loadtxt("metz/known_drug-target_interaction_affinities_pKi__Metz_et_al.2011.txt")
    XD = np.loadtxt("metz/drug-drug_similarities_2D__Metz_et_al.2011.txt")
    XT = np.loadtxt("metz/target-target_similarities_WS_normalized__Metz_et_al.2011.txt")

    if not pct is None:
        n, m = XD.shape[0], XT.shape[0]
        p = np.sqrt(pct)
        s, t = int(p*n), int(p*m)
        Y = Y[:s, :t]
        XD = XD[:s, :s]
        XT = XT[:t, :t]

    rows, cols = np.indices(Y.shape)
    has_edge = ~np.isnan(Y)

    drug_inds = rows[has_edge]
    target_inds = cols[has_edge]
    Y = Y[has_edge]

    if binarize:
        Y = np.where(Y > 7.6, 1, -1)

    return XD, XT, Y, drug_inds, target_inds

def load_kernels(fn1, fn2, mlpk=False, cols=None):

    with zipfile.ZipFile('Merget_2967drugs/drug_kernels/%s.zip' % fn1, 'r') as archive:
        with archive.open(fn1, mode='r') as file1:
            K1 = np.loadtxt(file1)

    with zipfile.ZipFile('Merget_2967drugs/drug_kernels/%s.zip' % fn2, 'r') as archive:
        with archive.open(fn2, mode='r') as file2:
            K2 = np.loadtxt(file2)

    if cols is not None:
        K1 = K1[:cols, :cols]
        K2 = K2[:cols, :cols]

    assert K1.shape == K2.shape
    assert (K1 == K1.T).all()
    assert (K2 == K2.T).all()

    return(K1, K2)


def load_merget(fn1=None, fn2=None, binarize=True):

    if not fn1 is None:
        fns = fn1
    else:
        fns = [fn[:-4] for fn in os.listdir('Merget_2967drugs/drug_kernels/') if fn.endswith('.zip')]

    drug_kernels = {}
    for fn1 in fns:
        print(fn1)
        with zipfile.ZipFile('Merget_2967drugs/drug_kernels/%s.zip' % fn1, 'r') as archive:
            with archive.open(fn1, mode='r') as file1:
                K1 = np.loadtxt(file1)
                drug_kernels[fn1] = K1

    if not fn2 is None:
        fns = fn2
    else:
        fns = ['Kp_GO-BP-log__GaussianK_gamma_1e-04.txt',
               'Kp_GO-CC-log__GaussianK_gamma_0.0039.txt',
               'Kp_GO-MF-log__GaussianK_gamma_0.0034.txt',
               'Kp_GS-ATP_L10_Sp1.0_Sc1.0.txt',
               'Kp_GS-KINDOM_L10_Sp1.0_Sc1.0.txt',
               'Kp_GS-Uniprot_L10_Sp1.0_Sc1.0.txt',
               'Kp_SW-ATP_shrunk.txt',
               'Kp_SW-KIN_DOM.txt',
               'Kp_SW-Uniprotseq.txt']

    protein_kernels = {}
    with zipfile.ZipFile('Merget_2967drugs\protein_kernels\protein_kernels.zip', 'r') as archive:
        for fn2 in fns:
            print(fn2)
            with archive.open(fn2, mode='r') as file2:
                K2 = np.loadtxt(file2)
                protein_kernels[fn2] = K2

    Y = np.loadtxt('Merget_2967drugs/Merget_DTIs_2967com_226kin.txt')
    if binarize:
        missing = np.isnan(Y)
        Y = np.where(Y > 7.6, 1.0, -1.0)
        Y[missing] = np.nan

    return(drug_kernels, protein_kernels, Y)


def load_heterodimers(kernel):
    unique_proteins = pd.read_csv('cyc_data/unique_proteins.csv', sep=';', index_col=0, header=None)[1].values
    Y = pd.read_csv("cyc_data/heterodimers.csv", sep=';').pivot(index='Protein A', columns='Protein B', values='Y')
    K = pd.read_csv("cyc_data/%s.csv" % kernel, sep=';', index_col=[0])

    Y = Y.reindex(index=unique_proteins, columns=unique_proteins).values
    K = K.reindex(index=unique_proteins, columns=unique_proteins).values

    rows, cols = np.indices(Y.shape)
    has_edge = ~np.isnan(Y)

    rows = rows[has_edge]
    cols = cols[has_edge]
    Y = Y[has_edge]

    print(K.shape, Y.shape, "(%d/%d)" % ((Y==1).sum(),(Y==-1).sum()), len(np.unique(rows)), len(np.unique(cols)))
    return(K, Y, rows, cols)


if __name__ == '__main__':
    X1, X2, rind, cind, Y = generate_chessboard(20,5)
    X, rind, cind, Y = generate_chessboard_singdom(20)
    XD, XT, Y, drug_inds, target_inds = load_metz()

    #print(XD.shape, XT.shape, Y.shape, drug_inds.shape, target_inds.shape)