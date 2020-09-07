import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from rlscore.kernel import GaussianKernel, LinearKernel
from rlscore.learner.cg_kron_rls import CGKronRLS
from rlscore.measure import cindex

from data import load_metz
from matvec import pko_linear, pko_poly2d, pko_kronecker, pko_symmetric, pko_mlpk, pko_cartesian
from validation import setting_kernels_heterogeneous, setting_kernels_heterogeneous_CV
from learner import RLScoreSaveAUC, RLScoreStop
from validation import K_to_dense
from sklearn.model_selection import ShuffleSplit


# RLScore implementation with early stopping determined by a validation set
def train_rlscore(KD, KT, Y, rows, cols, pko, setting=None, split_ratio=0.25, regparam=0.0001, maxiter=300):

    iterations = maxiter

    if not setting is None:

        kernels = setting_kernels_heterogeneous(KD, KT, Y, rows, cols, split_ratio, setting=setting)
        KD_inner, KT_inner, Y_inner, rows_inner, cols_inner = kernels['Train']
        KD_validation, KT_validation, Y_validation, rows_validation, cols_validation = kernels['Test']
        #print("Labelled pairs: Train %d, Inner %d, Validation %d" %(len(Y_train), len(Y_inner), len(Y_validation)))

        pko_inner = pko(KD_inner, KT_inner, rows_inner, cols_inner, rows_inner, cols_inner)
        pko_validation = pko(KD_validation, KT_validation, rows_validation, cols_validation, rows_inner, cols_inner)
        check_stop = RLScoreStop(Y_validation, pko_validation)
        try:
            kronrls = CGKronRLS(Y=Y_inner, pko=pko_inner, regparam=regparam, maxiter=maxiter, callback=check_stop)
        except StopIteration:
            #print("Stopped after", check_stop.iterations)
            pass

        iterations = check_stop.iterations

    pko_train = pko(KD, KT, rows, cols, rows, cols)
    kronrls = CGKronRLS(Y=Y, pko=pko_train, regparam=regparam, maxiter=iterations)

    return(kronrls, iterations)


'''
def setting_indices(row_inds, col_inds, split_rows=1./3, split_cols=1./3, split_setting1=1./2):
    R, C = len(row_inds), len(col_inds)
    rows, cols = np.unique(row_inds), np.unique(col_inds)
    assert R == C

    # Split to setting 1 and not setting1 columns
    rows, cols = np.unique(row_inds), np.unique(col_inds)
    cv1 = ShuffleSplit(n_splits=1, test_size=split_cols, random_state=0)
    cv2 = ShuffleSplit(n_splits=1, test_size=split_rows, random_state=0)
    train_cols, test_cols = next(cv1.split(cols))
    train_cols, test_cols = cols[train_cols], cols[test_cols]
    train_rows, test_rows = next(cv2.split(rows))
    train_rows, test_rows = rows[train_rows], rows[test_rows]

    # Divide indices into setting1, setting2, setting3, setting 4
    setting1 = np.isin(row_inds, train_rows) & np.isin(col_inds, train_cols)
    setting2 = np.isin(row_inds, train_rows) & np.isin(col_inds, test_cols)
    setting3 = np.isin(row_inds, test_rows) & np.isin(col_inds, train_cols)
    setting4 = np.isin(row_inds, test_rows) & np.isin(col_inds, test_cols)

    # Split setting1 into train/test1 and setting1, setting2, setting 3 into test2, test3, test4
    cv3 = ShuffleSplit(n_splits=1, test_size=split_setting1, random_state=0)
    indices = np.arange(C)
    base = indices[setting1]
    train_samples, test1_samples = next(cv3.split(base))
    train = base[train_samples]
    test1 = base[test1_samples]
    test2 = indices[setting2]
    test3 = indices[setting3]
    test4 = indices[setting4]

    return(train, test1, test2, test3, test4)

# Split given kernel K into train and test kernel matrices in settings 1/2/3/4, where each setting
# has a training kernel K_train (n_train x n_train) and a test kernel K_test (n_test x n_train)
def setting_kernels(KD, KT, Y, row_inds, col_inds):
    train, test1, test2, test3, test4 = setting_indices(row_inds, col_inds, N)

    kernels = {}

    Y_train = Y[train]
    rows_train, cols_train = row_inds[train], col_inds[train]
    KD_train, rows, rows = K_to_dense(KD, rows_train, rows_train)
    KT_train, cols, cols = K_to_dense(KT, cols_train, cols_train)
    kernels['Train'] = (KD_train, KT_train, Y_train, rows, cols)

    for setting, test in zip(['Setting 1', 'Setting 2', 'Setting 3', 'Setting 4'], [test1, test2, test3, test4]):
        Y_test = Y[test]
        rows_test, cols_test = row_inds[test], col_inds[test]
        KD_test, rows1, rows2 = K_to_dense(KD, rows_test, rows_train)
        KT_test, cols1, cols2 = K_to_dense(KT, cols_test, cols_train)
        kernels[setting] = (KD_test, KT_test, Y_test, rows1, cols1)

    return(kernels)
'''

# Train / Validation AUC over iterations and regularization parameters in settings 1/2/3/4
def hyperparameters_save(fn='ki_kernels_hyperpameters2.csv', regparam=0.0001, maxiter=300):

    XD, XT, Y, drug_inds, target_inds = load_metz(binarize=True)
    print("Drugs %d, Targets %d, Pairs %d" % (len(XT), len(XD), len(Y)))

    drug_kernel = GaussianKernel(XD, gamma=10**-5)# LinearKernel(XD)
    target_kernel = GaussianKernel(XT, gamma=10**-5)# LinearKernel(XT)
    KD = drug_kernel.getKM(XD)
    KT = target_kernel.getKM(XT)

    data = []

    # Iterate over Setting 1, ..., Setting 4
    for setting, cv in [('Setting 1', 16), ('Setting 2', 16), ('Setting 3', 16), ('Setting 4', 4)]:
        print(setting)

        for i, (train, test) in enumerate(setting_kernels_heterogeneous_CV(KD, KT, Y, drug_inds, target_inds, cv, setting=setting)):
            print("Fold", i+1)

            KD_train, KT_train, Y_train, rows_train, cols_train = train
            KD_test, KT_test, Y_test, rows_test, cols_test = test
            print("\t Train Labeled pairs %d, Test labelled pairs %d" %(len(Y_train), len(Y_test)))

            # Train Kronecker kernel
            start = time.perf_counter()
            pko_train = pko_kronecker(KD_train, KT_train, rows_train, cols_train, rows_train, cols_train)
            pko_test = pko_kronecker(KD_test, KT_test, rows_test, cols_test, rows_train, cols_train)
            save_aucs = RLScoreSaveAUC([Y_train, Y_test], [pko_train, pko_test])
            kronrls = CGKronRLS(Y=Y_train, pko=pko_train, regparam=regparam, maxiter=maxiter, callback=save_aucs)
            end = time.perf_counter()
            print("\t Took %f seconds" %(end - start))

            # Save Setting, Regularization, CV fold, Test AUC
            aucs = [(setting, i+1, j+1, auc_train, auc_test) for j, (auc_train, auc_test) in enumerate(save_aucs.aucs)]
            data.extend(aucs)

    data = pd.DataFrame(data, columns=['Setting', 'Fold', 'Iteration', 'Train AUC', 'Test AUC'])
    data.to_csv('results/%s' % fn, sep=';', index=False)

    return(data)

def hyperparameters_load(fn='ki_kernels_hyperpameters2.csv'):

    df = pd.read_csv('results/%s' % fn, sep=';')

    # Plot AUC / iteration and regularization parameter
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, sharey=True)
    df_train = df.groupby(['Setting', 'Iteration'])['Train AUC'].mean().reset_index()
    df_train.pivot(index='Iteration', columns='Setting', values='Train AUC').plot(ax=ax1).set_title('Train AUC/Setting')
    df_test = df.groupby(['Setting', 'Iteration'])['Test AUC'].mean().reset_index()
    df_test.pivot(index='Iteration', columns='Setting', values='Test AUC').plot(ax=ax2).set_title('Test AUC/Setting')
    plt.show()


kernels = {
    'Gaussian': lambda X: GaussianKernel(X, gamma=10**-5),
    'Linear': lambda X: LinearKernel(X)
}

# Train / Validation AUC over iterations and regularization parameters in settings 1/2/3/4
def compare_save(fn='ki_kernels2.csv', regparam=0.0001, maxiter=300):

    XD, XT, Y, drug_inds, target_inds = load_metz(binarize=True)
    print("Drugs %d, Targets %d, Pairs %d" % (len(XT), len(XD), len(Y)))

    data = []
    for domain_kernel in ['Gaussian', 'Linear']:
        print(domain_kernel)

        kernel_f = kernels[domain_kernel]
        drug_kernel = kernel_f(XD)
        target_kernel = kernel_f(XT)
        KD = drug_kernel.getKM(XD)
        KT = target_kernel.getKM(XT)

        # Iterate over Setting 1, ..., Setting 4
        for setting, cv in [('Setting 1', 9), ('Setting 2', 9), ('Setting 3', 9), ('Setting 4', 3)]:
            print(setting)

            for i, (train, test) in enumerate(setting_kernels_heterogeneous_CV(KD, KT, Y, drug_inds, target_inds, cv, setting=setting)):
                print("Fold", i+1)


                for kernel, pko in zip(['Linear', 'Poly2D', 'Kronecker', 'Cartesian'],
                                          [pko_linear, pko_poly2d, pko_kronecker, pko_cartesian]):
                    print(kernel)


                    KD_train, KT_train, Y_train, rows_train, cols_train = train
                    KD_test, KT_test, Y_test, rows_test, cols_test = test
                    print("\t Train Labeled pairs %d, Test labelled pairs %d" %(len(Y_train), len(Y_test)))

                    # Train Kronecker kernel
                    start = time.perf_counter()
                    kronrls, iterations = train_rlscore(KD_train, KT_train, Y_train, rows_train, cols_train, pko, setting=setting, maxiter=maxiter)
                    print("\t Iterations: %d" % iterations)
                    #pko_train = pko(KD_train, KT_train, rows_train, cols_train, rows_train, cols_train)
                    #kronrls = CGKronRLS(Y=Y_train, pko=pko_train, regparam=regparam, maxiter=maxiter)
                    pko_test = pko(KD_test, KT_test, rows_test, cols_test, rows_train, cols_train)
                    Y_pred = kronrls.predict(pko=pko_test)
                    auc = cindex(Y_test, Y_pred)
                    end = time.perf_counter()
                    print("\t Took %f seconds with AUC %.2f" %(end - start, auc))

                    # Save Setting, Regularization, CV fold, Test AUC
                    aucs = [(domain_kernel, kernel, setting, i+1, auc)]
                    data.extend(aucs)
            print()
        print()
        print()

    data = pd.DataFrame(data, columns=['Kernel (domain)', 'Kernel (pairwise)', 'Setting', 'Fold', 'AUC'])
    data.to_csv('results/%s' % fn, sep=';', index=False)

    return(data)

def compare_load(fn='ki_kernels2.csv'):

    data = pd.read_csv('results/%s' % fn, sep=';')
    print(data)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for ax, (kernel, df) in zip(axs, data.groupby('Kernel (domain)')):
        # Plot max AUC / regularization parameter
        df_max = df.groupby(['Kernel (pairwise)', 'Setting'])['AUC'].mean().unstack(level=1)
        df_max.plot(kind='bar', ax=ax).set_title(kernel)
        plt.ylim(0.0, 1.0)
    axs[0].legend().set_visible(False)
    axs[1].legend(bbox_to_anchor=(1.05, 1.00))
    fig.suptitle('Kernel (domain)')
    plt.tight_layout()
    plt.show()

    tb = data.groupby(['Kernel (pairwise)', 'Kernel (domain)', 'Setting'])[' AUC'].mean().unstack(level=0).round(2)
    tb = tb[['Linear', 'Poly2D', 'Kronecker', 'Cartesian']]
    tb.to_csv('table_ki2.csv', sep=';')
    print(tb)



if __name__ == '__main__':
    #hyperparameters_save()
    hyperparameters_load()
    #data = compare_save()
    #compare_load()
