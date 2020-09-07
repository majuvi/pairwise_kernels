import numpy as np
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import time
import pandas as pd

from rlscore.learner.cg_kron_rls import CGKronRLS
from rlscore.measure import cindex

from data import load_kernels
from matvec import pko_linear, pko_poly2d, pko_kronecker, pko_symmetric, pko_mlpk
from validation import setting_kernels_homogeneous, setting1_split, setting2_split, setting3_split, setting4_split, setting_kernels
from learner import RLScoreSaveAUC, RLScoreStop
from cichonska_MLPK import train_rlscore

# Substitute K for both kernels where matvec assumes K1, K2
pko_linear_ = lambda K, rows1, cols1, rows2, cols2: pko_linear(K, K, rows1, cols1, rows2, cols2)
pko_poly2d_ = lambda K, rows1, cols1, rows2, cols2: pko_poly2d(K, K, rows1, cols1, rows2, cols2)
pko_kronecker_ = lambda K, rows1, cols1, rows2, cols2: pko_kronecker(K, K, rows1, cols1, rows2, cols2)

# Test regularization effect of Tikhonov regularization (lambda) with early stopping (maxiter)
# Get regularization lambda = 0.0001, 0.001, ..., 1000, 10000
# Iterate for i = 1, ..., maxiter on train data and evaluate AUC in test1, test2, test3, test4
def hyperparameters_save(fn='cichonska_kernels_hyperpameters.csv', regparams=(0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000), maxiter=300):

    print("loading kernels...")
    K1, K2 = load_kernels('Kd_Tanimoto-estate.txt', 'Kd_Tanimoto-circular.txt', mlpk=False, cols=200)
    print(K1.shape, K2.shape)
    print()

    rows, cols = np.indices(K1.shape)
    has_edge = ~np.isnan(K1)
    row_inds, col_inds = rows[has_edge], cols[has_edge]
    Y = K2[has_edge]

    data = []
    # Iterate over Setting 1, ..., Setting 4
    for setting, split_ratio in [('Setting 1', 0.25), ('Setting 2', 0.25), ('Setting 3', 0.25), ('Setting 4', 0.36)]:
        print("Setting ", setting)

        kernels = setting_kernels_homogeneous(K1, Y, row_inds, col_inds, split_ratio, setting=setting)

        K_train, Y_train, rows_train, cols_train = kernels['Train']
        K_test, Y_test, rows_test, cols_test = kernels['Test']
        print("Train Labeled pairs %d, Test labelled pairs %d" %(len(Y_train), len(Y_test)))

        pko_train = pko_kronecker(K_train, K_train, rows_train, cols_train, rows_train, cols_train)
        pko_test = pko_kronecker(K_test, K_test, rows_test, cols_test, rows_train, cols_train)

        for regparam in regparams:
            print(regparam)

            # Train Kronecker kernel
            start = time.perf_counter()
            save_aucs = RLScoreSaveAUC([Y_train, Y_test], [pko_train, pko_test])
            kronrls = CGKronRLS(Y=Y_train, pko=pko_train, regparam=regparam, maxiter=maxiter, callback=save_aucs)
            end = time.perf_counter()
            print("Training took %f seconds" %(end - start))

            # Save Setting, Regularization, Iteration, Training AUC, Test AUC
            aucs = [(setting, regparam, i+1, auc_train, auc_test) for i, (auc_train, auc_test) in enumerate(save_aucs.aucs)]
            data.extend(aucs)
        print("")

    data = pd.DataFrame(data, columns=['Setting', 'Regularization', 'Iteration', 'Train AUC', 'Test AUC'])
    data.to_csv('results/%s' % fn, sep=';', index=False)

    return(data)

def hyperparameters_load(fn='cichonska_kernels_hyperpameters.csv'):

    data = pd.read_csv('results/%s' % fn, sep=';')

    # Plot AUC / iteration for different settings
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1,4, sharey=True)
    data[data['Setting'] == 'Setting 1'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax1).set_title('Setting 1')
    data[data['Setting'] == 'Setting 2'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax2).set_title('Setting 2')
    data[data['Setting'] == 'Setting 3'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax3).set_title('Setting 3')
    data[data['Setting'] == 'Setting 4'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax4).set_title('Setting 4')

    # Plot max AUC / setting
    fig, ax = plt.subplots(1, 1)
    df_max = data.groupby(['Regularization', 'Setting'])['Test AUC'].max().unstack(level=1)
    df_max.plot(logx=True, ax=ax)
    plt.show()

# Compare different kernels on kernel1 (train) => kernel2 (Y) filling task
# For kernel in Linear, Poly2D, Kronecker, Symmetric, MLPK:
# For test in test1, test2, test3, test4:
#   Split train into train and validation, iterate until AUC stops decreasing => iterations
#   Iterate original train for iterations and evaluate AUC in test
def compare_save(fn='cichonska_kernels.csv', regparam=0.0001, maxiter=500):

    kernel_fns = [
        'Kd_Tanimoto-circular.txt',
        'Kd_Tanimoto-estate.txt',
        'Kd_Tanimoto-extended.txt',
        'Kd_Tanimoto-graph.txt',
        'Kd_Tanimoto-hybridization.txt',
        'Kd_Tanimoto-kr.txt',
        'Kd_Tanimoto-maccs.txt',
        'Kd_Tanimoto-pubchem.txt',
        'Kd_Tanimoto-shortestpath.txt',
        'Kd_Tanimoto-standard.txt'
    ]
    kernel_fns1 = kernel_fns[0:1]
    kernel_fns2 = kernel_fns

    data = []
    i = 1
    for kernel1_fn in kernel_fns1:
        for kernel2_fn in kernel_fns2:
            print("=== %s x %s (%d/%d) ===" % (kernel1_fn, kernel2_fn, i, len(kernel_fns1)*len(kernel_fns2)))
            K1, K2 = load_kernels(kernel1_fn, kernel2_fn, mlpk=False)

            rows, cols = np.indices(K1.shape)
            has_edge = ~np.isnan(K1)
            row_inds, col_inds = rows[has_edge], cols[has_edge]
            Y = K2[has_edge]

            print("Randomizing...", end='')
            kernels = setting_kernels(K1, Y, row_inds, col_inds, N=40000)
            print('done.')
            K_train, Y_train, rows_train, cols_train = kernels['Train']

            for kernel, pko in zip(['Linear', 'Poly2D', 'Kronecker', 'Symmetric', 'MLPK'],
                                      [pko_linear_, pko_poly2d_, pko_kronecker_, pko_symmetric, pko_mlpk]):
                print(kernel)
                # Iterate over Setting 1, ..., Setting 4
                for setting, split_ratio in [('Setting 1', 0.25), ('Setting 2', 0.25), ('Setting 3', 0.25), ('Setting 4', 0.36)]:
                    print("Setting ", setting)

                    kronrls, iterations = train_rlscore(K_train, Y_train, rows_train, cols_train, pko, setting, regparam=regparam, maxiter=maxiter)

                    # Compute AUC on test set
                    K_test, Y_test, rows_test, cols_test = kernels[setting]
                    pko_test = pko(K_test, rows_test, cols_test, rows_train, cols_train)
                    P = kronrls.predict(pko=pko_test)
                    auc = cindex(P, Y_test)
                    # Save results
                    print("\t RLScore: Iterations %d, AUC %.2f " % (iterations, auc))
                    data.append((kernel1_fn, kernel2_fn, kernel, setting, auc))
            print("")
            i += 1

    data = pd.DataFrame(data, columns=['Kernel (domain)', 'Kernel (Y)', 'Kernel (pairwise)', 'Setting', 'Test AUC'])
    data.to_csv('results/%s' % fn, sep=';', index=False)

    return(data)

def compare_load(fn='cichonska_kernels.csv'):

    data = pd.read_csv('results/%s' % fn, sep=';')


    for (kernel1, kernel2), df in data.groupby(['Kernel (domain)', 'Kernel (Y)']):
        print(kernel1, kernel2)
        fig, ax = plt.subplots(1, 1)
        print(df)
        # Plot max AUC / regularization parameter
        df = df.set_index(['Kernel (pairwise)', 'Setting'])['Test AUC'].unstack(level=1)
        df.plot(kind='bar', ax=ax).set_title("%s => %s" % (kernel1, kernel2))
        plt.ylim(0.0, 1.0)
        plt.show()


if __name__ == '__main__':
    #hyperparameters_save()
    #hyperparameters_load()
    compare_save()
    compare_load()

