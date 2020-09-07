import numpy as np
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import time
import pandas as pd

from rlscore.learner.cg_kron_rls import CGKronRLS
from rlscore.measure import cindex

from data import load_merget
from matvec import pko_linear, pko_poly2d, pko_kronecker, pko_cartesian
from validation import setting_kernels_heterogeneous, setting1_split, setting2_split, setting3_split, setting4_split
from learner import RLScoreSaveAUC, RLScoreStop

# Substitute K for both kernels where matvec assumes K1, K2
pko_linear_ = lambda K, rows1, cols1, rows2, cols2: pko_linear(K, K, rows1, cols1, rows2, cols2)
pko_poly2d_ = lambda K, rows1, cols1, rows2, cols2: pko_poly2d(K, K, rows1, cols1, rows2, cols2)
pko_kronecker_ = lambda K, rows1, cols1, rows2, cols2: pko_kronecker(K, K, rows1, cols1, rows2, cols2)

# Test regularization effect of Tikhonov regularization (lambda) with early stopping (maxiter)
# Get regularization lambda = 0.0001, 0.001, ..., 1000, 10000
# Iterate for i = 1, ..., maxiter on train data and evaluate AUC in test1, test2, test3, test4
def hyperparameters_save(fn='cichonska_kernels2_hyperpameters.csv', regparams=(0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000), maxiter=300):

    print("loading kernels...")
    drug_kernels, protein_kernels, Y = load_merget(['Kd_Tanimoto-circular.txt'], ['Kp_GS-ATP_L5_Sp4.0_Sc4.0.txt'])
    KD = drug_kernels['Kd_Tanimoto-circular.txt']
    KT = protein_kernels['Kp_GS-ATP_L5_Sp4.0_Sc4.0.txt']
    print(KD.shape, KT.shape, Y.shape)
    print()

    rows, cols = np.indices(Y.shape)
    has_edge = ~np.isnan(Y)
    row_inds, col_inds = rows[has_edge], cols[has_edge]
    Y = Y[has_edge]

    data = []
    # Iterate over Setting 1, ..., Setting 4
    for setting, split_ratio in [('Setting 1', 0.25), ('Setting 2', 0.25), ('Setting 3', 0.25), ('Setting 4', 0.36)]:
        print("Setting ", setting)

        kernels = setting_kernels_heterogeneous(KD, KT, Y, row_inds, col_inds, split_ratio, setting=setting)

        KD_train, KT_train, Y_train, rows_train, cols_train = kernels['Train']
        KD_test, KT_test, Y_test, rows_test, cols_test = kernels['Test']
        print("Train Labeled pairs %d, Test labelled pairs %d" %(len(Y_train), len(Y_test)))

        pko_train = pko_kronecker(KD_train, KT_train, rows_train, cols_train, rows_train, cols_train)
        pko_test = pko_kronecker(KD_test, KT_test, rows_test, cols_test, rows_train, cols_train)

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

def hyperparameters_load(fn='cichonska_kernels2_hyperpameters.csv'):

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

# Compare different kernels on kernel_drug, kernel_protein (train) => interaction (Y) task
# For kernel in Linear, Poly2D, Kronecker, Symmetric, MLPK:
# For test in test1, test2, test3, test4:
#   Split train into train and validation, evaluate AUC in validation
def compare_save(fn='cichonska_kernels2b.csv', regparam=0.0001, maxiter=300):

    print("loading kernels...")
    #drug_kernels, protein_kernels, Y = load_merget()

    drugs = ['Kd_Tanimoto-shortestpath.txt', 'Kd_Tanimoto-circular.txt', 'Kd_Tanimoto-kr.txt']
    cells = ['Kp_GS-ATP_L5_Sp4.0_Sc4.0.txt', 'Kp_GS-KINDOM_L5_Sp4.0_Sc4.0.txt', 'Kp_GO-BP-log__GaussianK_gamma_1e-04.txt',
             'Kp_GO-CC-log__GaussianK_gamma_0.0039.txt', 'Kp_SW-KIN_DOM.txt', 'Kp_GS-Uniprot_L5_Sp3.0_Sc4.0.txt']
    drug_kernels, protein_kernels, Y = load_merget(drugs, cells)

    pairs = [('Kd_Tanimoto-shortestpath.txt', 'Kp_GS-ATP_L5_Sp4.0_Sc4.0.txt'),
             ('Kd_Tanimoto-circular.txt', 'Kp_GS-ATP_L5_Sp4.0_Sc4.0.txt'),
             ('Kd_Tanimoto-kr.txt', 'Kp_GS-ATP_L5_Sp4.0_Sc4.0.txt'),
             ('Kd_Tanimoto-circular.txt', 'Kp_GS-KINDOM_L5_Sp4.0_Sc4.0.txt'),
             ('Kd_Tanimoto-circular.txt', 'Kp_GO-BP-log__GaussianK_gamma_1e-04.txt'),
             ('Kd_Tanimoto-circular.txt', 'Kp_GO-CC-log__GaussianK_gamma_0.0039.txt'),
             ('Kd_Tanimoto-circular.txt', 'Kp_SW-KIN_DOM.txt'),
             ('Kd_Tanimoto-circular.txt', 'Kp_GS-Uniprot_L5_Sp3.0_Sc4.0.txt')]

    print("%d x %d" % (len(drug_kernels), len(protein_kernels)))
    print()

    rows, cols = np.indices(Y.shape)
    has_edge = ~np.isnan(Y)
    row_inds, col_inds = rows[has_edge], cols[has_edge]
    Y = Y[has_edge]

    data = []
    i = 1
    #for kernel1_fn, KD in drug_kernels.items():
    #    for kernel2_fn, KT in protein_kernels.items():
    for kernel1_fn, kernel2_fn in pairs:
        KD, KT = drug_kernels[kernel1_fn], protein_kernels[kernel2_fn]

        print("=== %s x %s (%d/%d) ===" % (kernel1_fn, kernel2_fn, i, len(drug_kernels)*len(protein_kernels)))

        for kernel, pko in zip(['Linear', 'Poly2D', 'Kronecker', 'Cartesian'],
                                  [pko_linear, pko_poly2d, pko_kronecker, pko_cartesian]):
            print(kernel)
            # Iterate over Setting 1, ..., Setting 4
            for setting, split_ratio in [('Setting 1', 0.25), ('Setting 2', 0.25), ('Setting 3', 0.25), ('Setting 4', 0.36)]:
                print("Setting ", setting)

                kernels = setting_kernels_heterogeneous(KD, KT, Y, row_inds, col_inds, split_ratio, setting=setting)

                KD_train, KT_train, Y_train, rows_train, cols_train = kernels['Train']
                KD_test, KT_test, Y_test, rows_test, cols_test = kernels['Test']
                print("Train Labeled pairs %d, Test labelled pairs %d" %(len(Y_train), len(Y_test)))

                pko_train = pko(KD_train, KT_train, rows_train, cols_train, rows_train, cols_train)
                pko_test = pko(KD_test, KT_test, rows_test, cols_test, rows_train, cols_train)

                # Train Kronecker kernel
                start = time.perf_counter()
                save_aucs = RLScoreSaveAUC([Y_train, Y_test], [pko_train, pko_test])
                kronrls = CGKronRLS(Y=Y_train, pko=pko_train, regparam=regparam, maxiter=maxiter, callback=save_aucs)
                end = time.perf_counter()
                print("Training took %f seconds" %(end - start))

                # Save Setting, Regularization, Iteration, Training AUC, Test AUC
                aucs = [(kernel1_fn, kernel2_fn, kernel, setting, i+1, auc_train, auc_test) for i, (auc_train, auc_test) in enumerate(save_aucs.aucs)]
                data.extend(aucs)
        print("")
        i += 1

    data = pd.DataFrame(data, columns=['Kernel (drug)', 'Kernel (protein)', 'Kernel (pairwise)', 'Setting', 'Iteration', 'Train AUC', 'Test AUC'])
    data.to_csv('results/%s' % fn, sep=';', index=False)

    return(data)

def compare_load(fn='cichonska_kernels2b.csv'):

    data = pd.read_csv('results/%s' % fn, sep=';')
    data = data[data['Kernel (drug)'].isin(['Kd_Tanimoto-shortestpath.txt', 'Kd_Tanimoto-circular.txt']) &
                (data['Kernel (protein)'] == 'Kp_GS-ATP_L5_Sp4.0_Sc4.0.txt')]
    data['Kernel (drug)'] = data['Kernel (drug)'].str.replace('.txt', '')
    data['Kernel (protein)'] = data['Kernel (protein)'].str.replace('.txt', '')
    #data.rename(columns=dict(zip(['Kernel (domain)', 'Kernel (Y)', 'Kernel (pairwise)'], ['Kernel (drug)', 'Kernel (protein)', 'Kernel (pairwise)'])), inplace=True)

    tb = data.groupby(['Kernel (pairwise)', 'Kernel (drug)', 'Kernel (protein)', 'Setting'])['Test AUC'].max().unstack(level=0).round(2)
    tb = tb[['Linear', 'Poly2D', 'Kronecker', 'Cartesian']]
    print(tb)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for ax, ((kernel1, kernel2), df) in zip(axs, data.groupby(['Kernel (drug)', 'Kernel (protein)'])):
        #print(kernel1, kernel2)
        # Plot max AUC / regularization parameter
        df_max = df.groupby(['Kernel (pairwise)', 'Setting'])['Test AUC'].max().unstack(level=1)
        #df_max.plot(kind='bar', ax=ax).set_title("%s\n%s" % (kernel1, kernel2))
        df_max.plot(kind='bar', ax=ax).set_title("%s" % kernel1)
        plt.ylim(0.0, 1.0)
    axs[0].legend().set_visible(False)
    axs[1].legend(bbox_to_anchor=(1.05, 1.00))
    fig.suptitle('Kernel (domain)')
    plt.tight_layout()
    plt.show()


def compare_load_all(fn='cichonska_kernels2b.csv'):

    data = pd.read_csv('results/%s' % fn, sep=';')
    data['Kernel (drug)'] = data['Kernel (drug)'].str.replace('.txt', '')
    data['Kernel (protein)'] = data['Kernel (protein)'].str.replace('.txt', '')
    #data.rename(columns=dict(zip(['Kernel (domain)', 'Kernel (Y)', 'Kernel (pairwise)'], ['Kernel (drug)', 'Kernel (protein)', 'Kernel (pairwise)'])), inplace=True)

    tb = data.groupby(['Kernel (pairwise)', 'Kernel (drug)', 'Kernel (protein)', 'Setting'])['Test AUC'].max().unstack(level=0).round(2)
    tb = tb[['Linear', 'Poly2D', 'Kronecker', 'Cartesian']]
    #tb.to_csv('table_kernels2.csv', sep=';')
    #print(tb)

    for (kernel1, kernel2), df in data.groupby(['Kernel (drug)', 'Kernel (protein)']):
        #print(kernel1, kernel2)
        fig, ax = plt.subplots(1, 1)
        # Plot max AUC / regularization parameter
        df_max = df.groupby(['Kernel (pairwise)', 'Setting'])['Test AUC'].max().unstack(level=1)
        df_max.plot(kind='bar', ax=ax).set_title("%s\n%s" % (kernel1, kernel2))
        #df_max.plot(kind='bar', ax=ax).set_title("%s" % kernel1)
        plt.ylim(0.0, 1.0)
        #ax.legend().set_visible(False)
        ax.legend(bbox_to_anchor=(1.05, 1.00))
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    #hyperparameters_save()
    #hyperparameters_load()
    compare_save()
    compare_load()
    #compare_load_all()

