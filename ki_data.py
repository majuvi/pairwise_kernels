import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from rlscore.kernel import GaussianKernel, LinearKernel
from rlscore.learner.cg_kron_rls import CGKronRLS
from rlscore.measure import cindex

from data import load_metz
from matvec import pko_linear, pko_poly2d, pko_kronecker, pko_symmetric, pko_mlpk, pko_cartesian
from validation import setting1_split, setting2_split, setting3_split, setting4_split, setting_kernels_heterogeneous
from learner import RLScoreSaveAUC
from matplotlib.legend_handler import HandlerTuple


# Train / Validation AUC over iterations and regularization parameters in settings 1/2/3/4
def hyperparameters_save(fn='ki_kernels_hyperpameters.csv', regparams=(0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000), maxiter=300):

    XD, XT, Y, drug_inds, target_inds = load_metz(binarize=True)
    print("Drugs %d, Targets %d, Pairs %d" % (len(XT), len(XD), len(Y)))

    drug_kernel = GaussianKernel(XD, gamma=10**-5)# LinearKernel(XD)
    target_kernel = GaussianKernel(XT, gamma=10**-5)# LinearKernel(XT)
    KD = drug_kernel.getKM(XD)
    KT = target_kernel.getKM(XT)

    data = []
    # Iterate over Setting 1, ..., Setting 4
    for setting, split_ratio in [('Setting 1', 0.25), ('Setting 2', 0.25), ('Setting 3', 0.25), ('Setting 4', 0.36)]:
        print("Setting ", setting)

        datas = setting_kernels_heterogeneous(KD, KT, Y, drug_inds, target_inds, split_ratio, setting=setting)

        KD_train, KT_train, Y_train, rows_train, cols_train = datas['Train']
        KD_test, KT_test, Y_test, rows_test, cols_test = datas['Test']
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

def hyperparameters_load(fn='ki_kernels_hyperpameters.csv'):

    df = pd.read_csv('results/%s' % fn, sep=';')

    # Plot AUC / iteration and regularization parameter
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    df[df['Setting'] == 'Setting 1'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax1).set_title('Setting 1')
    df[df['Setting'] == 'Setting 2'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax2).set_title('Setting 2')
    df[df['Setting'] == 'Setting 3'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax3).set_title('Setting 3')
    df[df['Setting'] == 'Setting 4'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax4).set_title('Setting 4')
    plt.show()

    # Plot AUC / iteration given certain regularization parameter
    ax1 = plt.subplot(131)
    ax2 = plt.subplot(132, sharex=ax1, sharey=ax1)
    ax3 = plt.subplot(133, sharey=ax1)
    df[df['Regularization'] == 0.0001].pivot(index='Iteration', columns='Setting', values='Train AUC').plot(ax=ax1)
    df[df['Regularization'] == 0.0001].pivot(index='Iteration', columns='Setting', values='Test AUC').plot(ax=ax2)
    # Plot max AUC / regularization parameter
    df_max = df.groupby(['Regularization', 'Setting'])['Test AUC'].max().unstack(level=1)
    df_min = df.groupby(['Regularization', 'Setting'])['Test AUC'].last().unstack(level=1)
    l1, = ax3.plot(df_max['Setting 1'].index, df_max['Setting 1'], linestyle='solid')
    l2, = ax3.plot(df_max['Setting 2'].index, df_max['Setting 2'], linestyle='solid')
    l3, = ax3.plot(df_max['Setting 3'].index, df_max['Setting 3'], linestyle='solid')
    l4, = ax3.plot(df_max['Setting 4'].index, df_max['Setting 4'], linestyle='solid')
    ax3.set_prop_cycle(None)
    ll1, = ax3.plot(df_min['Setting 1'].index, df_min['Setting 1'], linestyle='dotted')
    ll2, = ax3.plot(df_min['Setting 2'].index, df_min['Setting 2'], linestyle='dotted')
    ll3, = ax3.plot(df_min['Setting 3'].index, df_min['Setting 3'], linestyle='dotted')
    ll4, = ax3.plot(df_min['Setting 4'].index, df_min['Setting 4'], linestyle='dotted')
    ax3.set_xscale('log')

    ax1.set_ylabel('AUC')
    ax1.set_title(r'Train ($\lambda = 10^{-4}$)')
    ax1.set_xticks([0,100,200,300], minor=False)
    ax1.set_xticks([50,150,250], minor=True)
    ax1.legend().set_visible(False)
    plt.setp(ax1.get_xminorticklabels(), visible=False)
    ax2.set_title(r'Validation ($\lambda = 10^{-4}$)')
    ax2.set_xticks([0,100,200,300], minor=False)
    ax2.set_xticks([50,150,250], minor=True)
    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=4, handletextpad=0.1, labelspacing=0.1, handlelength=0.7)
    plt.setp(ax1.get_xminorticklabels(), visible=False)
    ax3.set_title('Validation')
    ax3.set_xlabel(r'$\lambda$')
    ax3.set_xticks((0.0001,0.01,1.0,100,10000), minor=False)
    ax3.set_xticks((0.001,0.1,10,1000,), minor=True)
    ax3.legend([(l1, l2, l3, l4), (ll1, ll2, ll3, ll4)], ['Early Stopping', 'Full solution'], scatterpoints=1,
               numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)}, loc='lower left')
    plt.setp(ax3.get_xminorticklabels(), visible=False)
    #plt.suptitle('AUC per iteration and the effect of early stopping')
    plt.show()


kernels = {
    'Gaussian': lambda X: GaussianKernel(X, gamma=10**-5),
    'Linear': lambda X: LinearKernel(X)
}

# Train / Validation AUC over iterations and regularization parameters in settings 1/2/3/4
def compare_save(fn='ki_kernels.csv', regparam=0.0001, maxiter=300):

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

        for kernel, pko in zip(['Linear', 'Poly2D', 'Kronecker', 'Cartesian'],
                                  [pko_linear, pko_poly2d, pko_kronecker, pko_cartesian]):
            print(kernel)

            # Iterate over Setting 1, ..., Setting 4
            for setting, split_ratio in [('Setting 1', 0.25), ('Setting 2', 0.25), ('Setting 3', 0.25), ('Setting 4', 0.36)]:
                print(setting)

                datas = setting_kernels_heterogeneous(KD, KT, Y, drug_inds, target_inds, split_ratio, setting=setting)

                KD_train, KT_train, Y_train, rows_train, cols_train = datas['Train']
                KD_test, KT_test, Y_test, rows_test, cols_test = datas['Test']
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
                aucs = [(domain_kernel, kernel, setting, i+1, auc_train, auc_test) for i, (auc_train, auc_test) in enumerate(save_aucs.aucs)]
                data.extend(aucs)
            print()
        print()
        print()

    data = pd.DataFrame(data, columns=['Kernel (domain)', 'Kernel (pairwise)', 'Setting', 'Iteration', 'Train AUC', 'Test AUC'])
    data.to_csv('results/%s' % fn, sep=';', index=False)

    return(data)

def compare_load(fn='ki_kernels.csv'):

    data = pd.read_csv('results/%s' % fn, sep=';')
    #print(data)

    fig, axs = plt.subplots(1, 2, sharex=True, sharey=True)
    for ax, (kernel, df) in zip(axs, data.groupby('Kernel (domain)')):
        # Plot max AUC / regularization parameter
        df_max = df.groupby(['Kernel (pairwise)', 'Setting'])['Test AUC'].max().unstack(level=1)
        df_max.plot(kind='bar', ax=ax).set_title(kernel)
        plt.ylim(0.0, 1.0)
    axs[0].legend().set_visible(False)
    axs[1].legend(bbox_to_anchor=(1.05, 1.00))
    fig.suptitle('Kernel (domain)')
    plt.tight_layout()
    plt.show()

    tb = data.groupby(['Kernel (pairwise)', 'Kernel (domain)', 'Setting'])['Test AUC'].max().unstack(level=0).round(2)
    tb = tb[['Linear', 'Poly2D', 'Kronecker', 'Cartesian']]
    tb.to_csv('table_ki.csv', sep=';')
    print(tb)



if __name__ == '__main__':
    hyperparameters_save()
    hyperparameters_load()
    compare_save()
    compare_load()
