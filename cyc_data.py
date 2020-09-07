import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from rlscore.learner.cg_kron_rls import CGKronRLS
from rlscore.measure import cindex

from data import load_heterodimers
from matvec import pko_linear, pko_poly2d, pko_kronecker, pko_cartesian, pko_symmetric, pko_mlpk
from validation import setting1_split, setting2_split, setting3_split, setting4_split, setting_kernels_homogeneous
from learner import RLScoreSaveAUC

# Substitute K for both kernels where matvec assumes K1, K2
pko_linear_ = lambda K, rows1, cols1, rows2, cols2: pko_linear(K, K, rows1, cols1, rows2, cols2)
pko_poly2d_ = lambda K, rows1, cols1, rows2, cols2: pko_poly2d(K, K, rows1, cols1, rows2, cols2)
pko_kronecker_ = lambda K, rows1, cols1, rows2, cols2: pko_kronecker(K, K, rows1, cols1, rows2, cols2)
pko_cartesian_ = lambda K, rows1, cols1, rows2, cols2: pko_cartesian(K, K, rows1, cols1, rows2, cols2)

'''
# Protein A, Protein B, heterodimer status
pairs = pd.read_csv('heterodimers.csv', sep=';', index=False)

# Protein list
unique_proteins = pd.read_csv('unique_proteins.csv', sep=';').values

# phi_domain(P), phi_location(P), phi_genome(P)
domain_map = pd.read_csv('protein_domain_map.csv', sep=';', index_col=0)
location_map = pd.read_csv('protein_location_map.csv', sep=';', index_col=0)
genome_map = pd.read_csv('protein_genome_map.csv', sep=';', index_col=0)
domain_map = domain_map.reindex(index=unique_proteins, fill_value=0)
location_map = location_map.reindex(index=unique_proteins, fill_value=0)

# Domain kernel (min, norm, minmax)
domain_min = pd.read_csv('K_domain_min.csv', sep=';')
domain_norm = pd.read_csv('K_domain_norm.csv', sep=';')
domain_minmax = pd.read_csv('K_domain_minmax.csv', sep=';')

# Location kernel (min, norm, minmax)
location_min = pd.read_csv('K_location_min.csv', sep=';')
location_norm = pd.read_csv('K_location_norm.csv', sep=';')
location_minmax = pd.read_csv('K_location_minmax.csv', sep=';')

# Genome kernel (min, norm, minmax)
genome_min = pd.read_csv('K_genome_min.csv', sep=';')
genome_norm = pd.read_csv('K_genome_norm.csv', sep=';')
genome_minmax = pd.read_csv('K_genome_minmax.csv', sep=';')

for kernel in (domain_min, domain_norm, domain_minmax,
               location_min, location_norm, location_minmax,
               genome_min, genome_norm, genome_minmax):
    print kernel.shape
'''

# Train / Validation AUC over iterations and regularization parameters in settings 1/2/3/4
def hyperparameters_save(fn='cyc_kernels_hyperpameters.csv', regparams=(0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100, 1000, 10000), maxiter=300):

    K, Y, row_inds, col_inds = load_heterodimers('K_domain_norm')

    data = []
    # Iterate over Setting 1, ..., Setting 4
    for setting, split_ratio in [('Setting 1', 0.25), ('Setting 2', 0.25), ('Setting 3', 0.25), ('Setting 4', 0.36)]:
        print("Setting ", setting)

        kernels = setting_kernels_homogeneous(K, Y, row_inds, col_inds, split_ratio, setting=setting)

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

def hyperparameters_load(fn='cyc_kernels_hyperpameters.csv'):

    df = pd.read_csv('results/%s' % fn, sep=';')

    # Plot AUC / iteration and regularization parameter
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, sharey=True)
    df[df['Setting'] == 'Setting 1'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax1).set_title('Setting 1')
    df[df['Setting'] == 'Setting 2'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax2).set_title('Setting 2')
    df[df['Setting'] == 'Setting 3'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax3).set_title('Setting 3')
    df[df['Setting'] == 'Setting 4'].pivot(index='Iteration', columns='Regularization', values='Test AUC').plot(ax=ax4).set_title('Setting 4')
    plt.show()

    ## Plot AUC / iteration given certain regularization parameter
    #fig, ax = plt.subplots(1, 1)
    #df[df['Regularization'] == 0.0001].pivot(index='Iteration', columns='Setting', values='Test AUC').plot(ax=ax).set_title('AUC given regularization = 0.0001')
    #plt.show()

    # Plot max AUC / regularization parameter
    fig, ax = plt.subplots(1, 1)
    df_max = df.groupby(['Regularization', 'Setting'])['Test AUC'].max().unstack(level=1)
    df_max.plot(logx=True, ax=ax).set_title('Maximum AUC over iterations')
    plt.show()


# Train / Validation AUC over iterations and regularization parameters in settings 1/2/3/4
def compare_save(fn='cyc_kernels.csv', regparam=0.0001, maxiter=100):

    data = []
    for domain_kernel in ['K_domain_minmax', #'K_domain_min', 'K_domain_norm',
                          'K_location_minmax', #'K_location_min', 'K_location_norm',
                          'K_genome_minmax', #'K_genome_min', 'K_genome_norm'
                          ]:
        print(domain_kernel)
        K, Y, row_inds, col_inds = load_heterodimers(domain_kernel)

        for kernel, pko in zip(['Linear', 'Poly2D', 'Kronecker', 'Cartesian', 'Symmetric', 'MLPK'],
                                  [pko_linear_, pko_poly2d_, pko_kronecker_, pko_cartesian_, pko_symmetric, pko_mlpk]):
            print(kernel)

            # Iterate over Setting 1, ..., Setting 4
            for setting, split_ratio in [('Setting 1', 0.25), ('Setting 2', 0.25), ('Setting 3', 0.25), ('Setting 4', 0.36)]:
                print(setting)

                kernels = setting_kernels_homogeneous(K, Y, row_inds, col_inds, split_ratio, setting=setting)

                K_train, Y_train, rows_train, cols_train = kernels['Train']
                K_test, Y_test, rows_test, cols_test = kernels['Test']
                #print("Train Labeled pairs %d, Test labelled pairs %d" %(len(Y_train), len(Y_test)))

                pko_train = pko(K_train, rows_train, cols_train, rows_train, cols_train)
                pko_test = pko(K_test, rows_test, cols_test, rows_train, cols_train)

                # Train Kronecker kernel
                start = time.perf_counter()
                save_aucs = RLScoreSaveAUC([Y_train, Y_test], [pko_train, pko_test])
                kronrls = CGKronRLS(Y=Y_train, pko=pko_train, regparam=regparam, maxiter=maxiter, callback=save_aucs)
                end = time.perf_counter()
                print("Training took %f seconds" %(end - start))

                # Save Setting, Regularization, Iteration, Training AUC, Test AUC
                aucs = [(domain_kernel, kernel, setting, i+1, auc_train, auc_test) for i, (auc_train, auc_test) in enumerate(save_aucs.aucs)]
                data.extend(aucs)
        print("")

    data = pd.DataFrame(data, columns=['Kernel (domain)', 'Kernel (pairwise)', 'Setting', 'Iteration', 'Train AUC', 'Test AUC'])
    data.to_csv('results/%s' % fn, sep=';', index=False)

    return(data)

def compare_load(fn='cyc_kernels.csv'):

    data = pd.read_csv('results/%s' % fn, sep=';')
    data['Kernel (domain)'] = data['Kernel (domain)'].map({'K_domain_minmax': 'K_domain', 'K_location_minmax': 'K_location', 'K_genome_minmax': 'K_genome'})
    #print(data)

    fig, axs = plt.subplots(1, 3, sharex=True, sharey=True)
    for ax, (kernel, df) in zip(axs, data.groupby('Kernel (domain)')):
        # Plot max AUC / regularization parameter
        df_max = df.groupby(['Kernel (pairwise)', 'Setting'])['Test AUC'].max().unstack(level=1)
        df_max.plot(kind='bar', ax=ax).set_title(kernel)
        plt.ylim(0.0, 1.0)
    axs[0].legend().set_visible(False)
    axs[1].legend().set_visible(False)
    axs[2].legend(bbox_to_anchor=(1.05, 1.00))
    fig.suptitle('Kernel (domain)')
    plt.tight_layout()
    plt.show()

    tb = data.groupby(['Kernel (pairwise)', 'Kernel (domain)', 'Setting'])['Test AUC'].max().unstack(level=0).round(2)
    tb = tb[['Linear', 'Poly2D', 'Kronecker', 'Cartesian', 'Symmetric', 'MLPK']]
    #tb.to_csv('table_cyc.csv', sep=';')
    print(tb)



if __name__ == '__main__':
    #hyperparameters_save()
    #hyperparameters_load()
    compare_save()
    compare_load()



