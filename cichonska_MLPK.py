import numpy as np
import matplotlib
#matplotlib.use('Qt4Agg')
import matplotlib.pyplot as plt
import time
import pandas as pd
import tracemalloc

from scipy.sparse.linalg import LinearOperator, minres
from rlscore.learner.cg_kron_rls import CGKronRLS
from rlscore.measure import cindex
from rlscore.measure.measure_utilities import UndefinedPerformance

from data import load_kernels
from matvec import linear_kernel, poly2d_kernel, kronecker_kernel, symmetric_kernel, mlpk_kernel
from matvec import mv_linear, mv_poly2d, mv_kronecker, mv_symmetric, mv_mlpk
from matvec import pko_linear, pko_poly2d, pko_kronecker, pko_symmetric, pko_mlpk
from validation import setting_kernels, setting1_split, setting2_split, setting3_split, setting4_split, setting_kernels_homogeneous
from learner import CheckStop, RLScoreStop, StopIteration

# Substitute K for both kernels where matvec assumes K1, K2
linear_kernel_ = lambda K, rows1, cols1, rows2, cols2: linear_kernel(K, K, rows1, cols1, rows2, cols2)
poly2d_kernel_ = lambda K, rows1, cols1, rows2, cols2: poly2d_kernel(K, K, rows1, cols1, rows2, cols2)
kronecker_kernel_ = lambda K, rows1, cols1, rows2, cols2: kronecker_kernel(K, K, rows1, cols1, rows2, cols2)
pko_linear_ = lambda K, rows1, cols1, rows2, cols2: pko_linear(K, K, rows1, cols1, rows2, cols2)
pko_poly2d_ = lambda K, rows1, cols1, rows2, cols2: pko_poly2d(K, K, rows1, cols1, rows2, cols2)
pko_kronecker_ = lambda K, rows1, cols1, rows2, cols2: pko_kronecker(K, K, rows1, cols1, rows2, cols2)
mv_kronecker_ = lambda K, rows1, cols1, rows2, cols2: mv_kronecker(K, K, rows1, cols1, rows2, cols2)


# RLScore implementation with early stopping determined by a validation set
def train_rlscore(K_train, Y_train, rows_train, cols_train, pko, setting=None, split_ratio=0.25, regparam=0.0001, maxiter=300):

    iterations = maxiter

    if not setting is None:

        kernels = setting_kernels_homogeneous(K_train, Y_train, rows_train, cols_train, split_ratio, setting=setting)
        K_inner, Y_inner, rows_inner, cols_inner = kernels['Train']
        K_validation, Y_validation, rows_validation, cols_validation = kernels['Test']
        #print("Labelled pairs: Train %d, Inner %d, Validation %d" %(len(Y_train), len(Y_inner), len(Y_validation)))

        pko_inner = pko(K_inner, rows_inner, cols_inner, rows_inner, cols_inner)
        pko_validation = pko(K_validation, rows_validation, cols_validation, rows_inner, cols_inner)
        check_stop = RLScoreStop(Y_validation, pko_validation)
        try:
            kronrls = CGKronRLS(Y=Y_inner, pko=pko_inner, regparam=regparam, maxiter=maxiter, callback=check_stop)
        except StopIteration:
            #print("Stopped after", check_stop.iterations)
            pass

        iterations = check_stop.iterations

    pko_train = pko(K_train, rows_train, cols_train, rows_train, cols_train)
    kronrls = CGKronRLS(Y=Y_train, pko=pko_train, regparam=regparam, maxiter=iterations)

    return(kronrls, iterations)

# Naive Kernel implementation with early stopping determined by a validation set
def train_kernel(K_train, Y_train, rows_train, cols_train, kernel, setting=None, split_ratio=0.25, regparam=0.0001, maxiter=300):

    iterations = maxiter

    if not setting is None:

        kernels = setting_kernels_homogeneous(K_train, Y_train, rows_train, cols_train, split_ratio, setting=setting)
        K_inner, Y_inner, rows_inner, cols_inner = kernels['Train']
        K_validation, Y_validation, rows_validation, cols_validation = kernels['Test']
        #print("Labelled pairs: Train %d, Inner %d, Validation %d" %(len(Y_train), len(Y_inner), len(Y_validation)))

        K_inner_pairs = kernel(K_inner, rows_inner, cols_inner, rows_inner, cols_inner)
        G_inner_pairs = LinearOperator(K_inner_pairs.shape, matvec = lambda v: np.dot(K_inner_pairs, v) + regparam * v, dtype = np.float64)
        K_validation_pairs = kernel(K_validation, rows_validation, cols_validation, rows_inner, cols_inner)
        G_validation_pairs = LinearOperator(K_validation_pairs.shape, matvec = lambda v: np.dot(K_validation_pairs, v), dtype = np.float64) # Predict with G_validation(A)
        check_stop = CheckStop(Y_validation, G_validation_pairs)
        try:
            A = minres(G_inner_pairs, Y_inner, maxiter=iterations, tol=1e-20, callback=check_stop)[0]
        except StopIteration:
            #print("Stopped after", check_stop.iterations)
            pass

        iterations = check_stop.iterations

    K_train_pairs = kernel(K_train, rows_train, cols_train, rows_train, cols_train)
    G_train_pairs = LinearOperator(K_train_pairs.shape, matvec = lambda v: np.dot(K_train_pairs, v) + regparam * v, dtype = np.float64)
    A = minres(G_train_pairs, Y_train, maxiter=iterations, tol=1e-20)[0]

    return(A, iterations)


# Get data sets N = 1000, 2000, ...
# For data set: N => data, test1, test2, test3, test4
# For data set: data => train, validation
# For kernel: linear, poly2d, kronecker, kronecker symmetric, MLPK
#   Iterate train, validation until no improvement in validation for 10 iterations => iterations
#   Iterate data for iterations in train => model, cpu, mem
#   Evaluate AUC in test1, test2, test3, test4
#   Save cpu usage, mem usage, etc.
def compare_save_old(fn='cichonska_MLPK.csv', regparam=0.0001, maxiter=300):

    print("loading kernels...")
    K1, K2 = load_kernels('Kd_Tanimoto-estate.txt', 'Kd_Tanimoto-circular.txt', mlpk=True)
    print(K1.shape, K2.shape)
    print()

    rows, cols = np.indices(K1.shape)
    has_edge = ~np.isnan(K1)
    row_inds, col_inds = rows[has_edge], cols[has_edge]
    Y = K2[has_edge]

    tests = []
    for N in 1000*2**np.arange(0,11): #11
        print("Samples", N)

        print("Randomizing...")
        kernels = setting_kernels(K1, Y, row_inds, col_inds, N)
        print()

        K_train, Y_train, rows_train, cols_train = kernels['Train']

        for kernel, naive, pko in zip(['Linear', 'Poly2D', 'Kronecker', 'Symmetric', 'MLPK'],
                                  [linear_kernel_, poly2d_kernel_, kronecker_kernel_, symmetric_kernel, mlpk_kernel],
                                  [pko_linear_, pko_poly2d_, pko_kronecker_, pko_symmetric, pko_mlpk]):
            print(kernel)
            for setting, split_ratio in [('Setting 1', 0.25), ('Setting 2', 0.25), ('Setting 3', 0.25), ('Setting 4', 0.36)]:
                print(setting)

                K_test, Y_test, rows_test, cols_test = kernels[setting]
                #print("K_train:", K_train.shape, len(np.unique(rows_train)), len(np.unique(cols_train)), len(Y_train))
                #print("K_test:", K_test.shape, len(np.unique(rows_test)), len(np.unique(cols_test)), len(Y_test))

                # Start benchmark
                tracemalloc.start()
                time_start = time.perf_counter()
                # "Load" data and fit RLscore using the training set
                K_train, Y_train, rows_train, cols_train = K_train.copy(), Y_train.copy(), rows_train.copy(), cols_train.copy()
                kronrls, iterations = train_rlscore(K_train, Y_train, rows_train, cols_train, pko, setting, regparam=regparam, maxiter=maxiter)
                # End benchmark
                time_train = time.perf_counter() - time_start
                mem_train = tracemalloc.get_traced_memory()[1]/10**9
                tracemalloc.stop()
                # Compute AUC on test set
                pko_test = pko(K_test, rows_test, cols_test, rows_train, cols_train)
                P = kronrls.predict(pko=pko_test)
                auc = cindex(P, Y_test)
                # Save results
                print("\t RLScore: Iterations %d, AUC %.2f " % (iterations, auc))
                tests.append((N, kernel, 'RLScore', setting, time_train, mem_train, iterations, auc))

                if N <= 32000:
                    # Start benchmark
                    tracemalloc.start()
                    time_start = time.perf_counter()
                    # "Load" data and fit RLscore using the training set
                    K_train, Y_train, rows_train, cols_train = K_train.copy(), Y_train.copy(), rows_train.copy(), cols_train.copy()
                    A, iterations = train_kernel(K_train, Y_train, rows_train, cols_train, naive, setting, regparam=regparam, maxiter=maxiter)
                    # End benchmark
                    time_train = time.perf_counter() - time_start
                    mem_train = tracemalloc.get_traced_memory()[1]/10**9
                    tracemalloc.stop()
                    ## Compute AUC on test set
                    P = pko_test.matvec(A)
                    auc = cindex(P, Y_test)
                    # Save results
                    print("\t Kernel: Iterations %d, AUC %.2f " % (iterations, auc))
                    tests.append((N, kernel, 'Kernel', setting, time_train, mem_train, iterations, auc))

            print()
        print()
        print()

    tests = pd.DataFrame(tests, columns=['N', 'Kernel', 'Method', 'Setting', 'Time', 'MEM', 'Iterations', 'AUC'])
    tests.to_csv('results/%s' % fn, sep=';', index=False)

    return(tests)

# For data set: data => train, validation
# For kernel: linear, poly2d, kronecker, kronecker symmetric, MLPK
#   Iterate train, validation until no improvement in validation for 10 iterations => iterations
#   Iterate data for iterations in train => model, cpu, mem
#   Evaluate AUC in test1, test2, test3, test4
#   Save cpu usage, mem usage, etc.
def compare_save(fn='cichonska_MLPKb.csv', regparam=0.0001, maxiter=300):

    print("loading kernels...")
    K1, K2 = load_kernels('Kd_Tanimoto-estate.txt', 'Kd_Tanimoto-circular.txt', mlpk=True)
    print(K1.shape, K2.shape)
    print()

    rows, cols = np.indices(K1.shape)
    has_edge = ~np.isnan(K1)
    row_inds, col_inds = rows[has_edge], cols[has_edge]
    Y = K2[has_edge]

    tests = []
    for N in 1000*2**np.arange(0,11): #11
        print("Samples", N)

        print("Randomizing...")
        kernels = setting_kernels(K1, Y, row_inds, col_inds, N)
        print()

        K_train, Y_train, rows_train, cols_train = kernels['Train']

        for kernel, naive, pko in zip(['Linear', 'Poly2D', 'Kronecker', 'Symmetric', 'MLPK'],
                                  [linear_kernel_, poly2d_kernel_, kronecker_kernel_, symmetric_kernel, mlpk_kernel],
                                  [pko_linear_, pko_poly2d_, pko_kronecker_, pko_symmetric, pko_mlpk]):
            print(kernel)
            for setting, split_ratio in [('Setting 1', 0.25), ('Setting 2', 0.25), ('Setting 3', 0.25), ('Setting 4', 0.36)]:
                print(setting)

                K_test, Y_test, rows_test, cols_test = kernels[setting]

                kernels2 = setting_kernels_homogeneous(K_train, Y_train, rows_train, cols_train, split_ratio, setting=setting)
                K_inner, Y_inner, rows_inner, cols_inner = kernels2['Train']
                K_validation, Y_validation, rows_validation, cols_validation = kernels2['Test']
                print("\t Labelled pairs: Train %d, Inner %d, Validation %d" % (len(Y_train), len(Y_inner), len(Y_validation)))

                pko_inner = pko(K_inner, rows_inner, cols_inner, rows_inner, cols_inner)
                pko_validation = pko(K_validation, rows_validation, cols_validation, rows_inner, cols_inner)
                check_stop = RLScoreStop(Y_validation, pko_validation)
                try:
                    kronrls = CGKronRLS(Y=Y_inner, pko=pko_inner, regparam=regparam, maxiter=maxiter, callback=check_stop)
                except StopIteration:
                    #print("Stopped after", check_stop.iterations)
                    pass
                iterations = check_stop.iterations
                print("\t Iterations %d" % iterations)

                # Start benchmark
                tracemalloc.start()
                time_start = time.perf_counter()
                # "Load" data and fit RLscore using the training set
                K_temp, Y_temp, rows_temp, cols_temp = K_train.copy(), Y_train.copy(), rows_train.copy(), cols_train.copy()
                pko_train = pko(K_temp, rows_temp, cols_temp, rows_temp, cols_temp)
                kronrls = CGKronRLS(Y=Y_temp, pko=pko_train, regparam=regparam, maxiter=iterations)
                # End benchmark
                time_train = time.perf_counter() - time_start
                mem_train = tracemalloc.get_traced_memory()[1]/10**9
                tracemalloc.stop()
                # Compute AUC on test set
                pko_test = pko(K_test, rows_test, cols_test, rows_train, cols_train)
                P = kronrls.predict(pko=pko_test)
                auc = cindex(P, Y_test)
                # Save results
                print("\t RLScore: AUC %.2f " % auc)
                tests.append((N, kernel, 'RLScore', setting, time_train, mem_train, iterations, auc))

                if N <= 32000:
                    # Start benchmark
                    tracemalloc.start()
                    time_start = time.perf_counter()
                    # "Load" data and fit RLscore using the training set
                    K_temp, Y_temp, rows_temp, cols_temp = K_train.copy(), Y_train.copy(), rows_train.copy(), cols_train.copy()
                    K_temp_pairs = naive(K_temp, rows_temp, cols_temp, rows_temp, cols_temp)
                    G_temp = LinearOperator(K_temp_pairs.shape, matvec = lambda v: np.dot(K_temp_pairs, v) + regparam * v, dtype = np.float64)
                    A = minres(G_temp, Y_temp, maxiter=iterations, tol=1e-20)[0]
                    # End benchmark
                    time_train = time.perf_counter() - time_start
                    mem_train = tracemalloc.get_traced_memory()[1]/10**9
                    tracemalloc.stop()
                    ## Compute AUC on test set
                    P = pko_test.matvec(A)
                    auc = cindex(P, Y_test)
                    # Save results
                    print("\t Kernel: AUC %.2f " % auc)
                    tests.append((N, kernel, 'Kernel', setting, time_train, mem_train, iterations, auc))

            print()
        print()
        print()

    tests = pd.DataFrame(tests, columns=['N', 'Kernel', 'Method', 'Setting', 'Time', 'MEM', 'Iterations', 'AUC'])
    tests.to_csv('results/%s' % fn, sep=';', index=False)

    return(tests)


def compare_load(fn='cichonska_MLPKb.csv'):

    tests = pd.read_csv('results/%s' % fn, sep=';')

    #tests[(tests['N'] == 1000*2**10) & (tests['Method'] =='RLScore')].pivot(index='Kernel', columns='Setting', values='AUC').plot(kind='bar').set_title('AUCs / kernel')
    #plt.show()

    print("AUC")
    print(pd.pivot_table(tests[tests['Method'] == 'RLScore'], index=['Kernel', 'Setting'], columns='N', values='AUC'))
    print(pd.pivot_table(tests[tests['Method'] == 'Kernel'], index=['Kernel', 'Setting'], columns='N', values='AUC'))
    print()
    print("Time")
    print(pd.pivot_table(tests[tests['Method'] == 'RLScore'], index=['Kernel', 'Setting'], columns='N', values='Time'))
    print(pd.pivot_table(tests[tests['Method'] == 'Kernel'], index=['Kernel', 'Setting'], columns='N', values='Time'))
    print()
    print("MEM")
    print(pd.pivot_table(tests[tests['Method'] == 'RLScore'], index=['Kernel', 'Setting'], columns='N', values='MEM'))
    print(pd.pivot_table(tests[tests['Method'] == 'Kernel'], index=['Kernel', 'Setting'], columns='N', values='MEM'))
    print()
    print("Iterations")
    print(pd.pivot_table(tests[tests['Method'] == 'RLScore'], index=['Kernel', 'Setting'], columns='N', values='Iterations'))
    print(pd.pivot_table(tests[tests['Method'] == 'Kernel'], index=['Kernel', 'Setting'], columns='N', values='Iterations'))
    print()

def compare_plot(fn='cichonska_MLPKb.csv'):

    tests = pd.read_csv('results/%s' % fn, sep=';')

    #tests[(tests['N'] == 1000*2**10) & (tests['Method'] =='RLScore')].pivot(index='Kernel', columns='Setting', values='AUC').plot(kind='bar').set_title('AUCs / kernel')
    #plt.show()

    fig = plt.figure()
    #fig.suptitle('Kernel Filling: AUC/Iterations/Time/MEM vs. Samples')

    ax11 = fig.add_subplot(4, 4, 1)
    ax12 = fig.add_subplot(4, 4, 2, sharey=ax11, sharex=ax11)
    ax13 = fig.add_subplot(4, 4, 3, sharey=ax11, sharex=ax11)
    ax14 = fig.add_subplot(4, 4, 4, sharey=ax11, sharex=ax11)
    #pd.pivot_table(tests[(tests['Method'] == 'RLScore')], index=['Setting', 'N'], columns='Kernel', values='AUC').to_csv('temp_auc.csv', sep=';')
    df1 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 1')], index='N', columns='Kernel', values='AUC')
    df2 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 2')], index='N', columns='Kernel', values='AUC')
    df3 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 3')], index='N', columns='Kernel', values='AUC')
    df4 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 4')], index='N', columns='Kernel', values='AUC')
    df1.plot(ax=ax11, logx=True)
    df2.plot(ax=ax12, logx=True)
    df3.plot(ax=ax13, logx=True)
    df4.plot(ax=ax14, logx=True)
    #fig.suptitle('Test AUC')
    ax11.set_ylabel('AUC')
    ax11.set_xlabel('')
    ax12.set_xlabel('')
    ax13.set_xlabel('')
    ax14.set_xlabel('')
    ax11.set_title('Setting 1')
    ax12.set_title('Setting 2')
    ax13.set_title('Setting 3')
    ax14.set_title('Setting 4')
    ax11.legend(loc='upper left', bbox_to_anchor=(0.5, 1.6), ncol=5, handletextpad=0.1, labelspacing=0.1, handlelength=0.7)
    ax12.legend().set_visible(False)
    ax13.legend().set_visible(False)
    ax14.legend().set_visible(False)
    ax11.set_yticks([0.4,0.5,0.6,0.7,0.8])
    ax11.set_ylim([0.45,0.8])
    #fig.subplots_adjust(hspace=0.25, wspace=0.05)

    ax21 = fig.add_subplot(4, 4, 5, sharex=ax11)
    ax22 = fig.add_subplot(4, 4, 6, sharey=ax21, sharex=ax11)
    ax23 = fig.add_subplot(4, 4, 7, sharey=ax21, sharex=ax11)
    ax24 = fig.add_subplot(4, 4, 8, sharey=ax21, sharex=ax11)
    #pd.pivot_table(tests[(tests['Method'] == 'RLScore')], index=['Setting', 'N'], columns='Kernel', values='Iterations').to_csv('temp_iterations.csv', sep=';')
    df1 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 1')], index='N', columns='Kernel', values='Iterations')
    df2 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 2')], index='N', columns='Kernel', values='Iterations')
    df3 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 3')], index='N', columns='Kernel', values='Iterations')
    df4 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 4')], index='N', columns='Kernel', values='Iterations')
    df1.plot(ax=ax21, logx=True)
    df2.plot(ax=ax22, logx=True)
    df3.plot(ax=ax23, logx=True)
    df4.plot(ax=ax24, logx=True)
    #fig.suptitle('Iterations')
    ax21.set_ylabel('Iterations')
    ax21.set_xlabel('')
    ax22.set_xlabel('')
    ax23.set_xlabel('')
    ax24.set_xlabel('')
    ax21.legend().set_visible(False)
    ax22.legend().set_visible(False)
    ax23.legend().set_visible(False)
    ax24.legend().set_visible(False)
    ax21.set_yticks([0,100,200,300])
    ax21.set_ylim([1, 300])
    #fig.subplots_adjust(hspace=0.25, wspace=0.05)

    ax31 = fig.add_subplot(4, 4, 9, sharex=ax11)
    ax32 = fig.add_subplot(4, 4, 10, sharey=ax31, sharex=ax11)
    ax33 = fig.add_subplot(4, 4, 11, sharey=ax31, sharex=ax11)
    ax34 = fig.add_subplot(4, 4, 12, sharey=ax31, sharex=ax11)
    ax41 = fig.add_subplot(4, 4, 13, sharex=ax11)
    ax42 = fig.add_subplot(4, 4, 14, sharey=ax41, sharex=ax11)
    ax43 = fig.add_subplot(4, 4, 15, sharey=ax41, sharex=ax11)
    ax44 = fig.add_subplot(4, 4, 16, sharey=ax41, sharex=ax11)
    for which, (ax1, ax2, ax3, ax4) in [('Time', (ax31, ax32, ax33, ax34)), ('MEM', (ax41, ax42, ax43, ax44))]:

        #pd.pivot_table(tests[(tests['Method'] == 'RLScore')], index=['Setting', 'N'], columns='Kernel', values=which).to_csv('temp_rlscore_%s.csv' % which, sep=';')
        df1 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 1')], index='N', columns='Kernel', values=which)
        df2 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 2')], index='N', columns='Kernel', values=which)
        df3 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 3')], index='N', columns='Kernel', values=which)
        df4 = pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 4')], index='N', columns='Kernel', values=which)
        df1.plot(ax=ax1, logx=True, logy=True)
        df2.plot(ax=ax2, logx=True, logy=True)
        df3.plot(ax=ax3, logx=True, logy=True)
        df4.plot(ax=ax4, logx=True, logy=True)

        #pd.pivot_table(tests[(tests['Method'] == 'Kernel')], index=['Setting', 'N'], columns='Kernel', values=which).to_csv('temp_kernel_%s.csv' % which, sep=';')
        df1 = pd.pivot_table(tests[(tests['Method'] == 'Kernel') & (tests['Setting'] == 'Setting 1')], index='N', columns='Kernel', values=which)
        df2 = pd.pivot_table(tests[(tests['Method'] == 'Kernel') & (tests['Setting'] == 'Setting 2')], index='N', columns='Kernel', values=which)
        df3 = pd.pivot_table(tests[(tests['Method'] == 'Kernel') & (tests['Setting'] == 'Setting 3')], index='N', columns='Kernel', values=which)
        df4 = pd.pivot_table(tests[(tests['Method'] == 'Kernel') & (tests['Setting'] == 'Setting 4')], index='N', columns='Kernel', values=which)
        ax1.set_prop_cycle(None)
        ax2.set_prop_cycle(None)
        ax3.set_prop_cycle(None)
        ax4.set_prop_cycle(None)
        df1.plot(ax=ax1, logx=True, logy=True, linestyle='--')
        df2.plot(ax=ax2, logx=True, logy=True, linestyle='--')
        df3.plot(ax=ax3, logx=True, logy=True, linestyle='--')
        df4.plot(ax=ax4, logx=True, logy=True, linestyle='--')

    ax31.set_yticks([10**-2, 10**0, 10**2, 10**4])
    ax41.set_yticks([10**-5, 10**-3, 10**-1, 10**1])
    ax41.set_ylim([10**-4, 16])

        #fig.suptitle('%s' % which)
    #fig.subplots_adjust(hspace=0.25, wspace=0.05)
    ax31.set_ylabel('Time')
    ax31.set_xlabel('')
    ax32.set_xlabel('')
    ax33.set_xlabel('')
    ax34.set_xlabel('')
    ax31.legend().set_visible(False)
    ax32.legend().set_visible(False)
    ax33.legend().set_visible(False)
    ax34.legend().set_visible(False)
    ax41.set_ylabel('MEM')
    ax41.set_xlabel('Samples')
    ax42.set_xlabel('Samples')
    ax43.set_xlabel('Samples')
    ax44.set_xlabel('Samples')
    ax41.legend().set_visible(False)
    ax42.legend().set_visible(False)
    ax43.legend().set_visible(False)
    ax44.legend().set_visible(False)


    ax11.set_xticks([1000, 10000, 100000, 1000000])
    #ax11.xaxis.set_tick_params(rotation=90)


    plt.show()



#print(pd.pivot_table(tests[(tests['Method'] == 'RLScore') & (tests['Setting'] == 'Setting 1')], index=['N', 'Setting'], columns='Kernel', values='AUC'))

# Test dense data set speeds
# Get data set: N = 1000, 2000, ... => train
# Iterate GVT/RLScore/Naive Kernel => cpu, mem, auc
def dense_save(fn='cichonska_MLPK_dense.csv', regparam = 0.0001, iterations=100):

    print("loading kernels...")
    K1, K2 = load_kernels('Kd_Tanimoto-estate.txt', 'Kd_Tanimoto-circular.txt', mlpk=True)
    print(K1.shape, K2.shape)
    print()

    rows, cols = np.indices(K1.shape)
    has_edge = ~np.isnan(K1)
    row_inds, col_inds = rows[has_edge], cols[has_edge]
    Y = K2[has_edge]

    tests = []
    for N in range(1000, 10000+1000, 1000):

        print("Randomizing...")
        kernels = setting_kernels(K1, Y, row_inds, col_inds, N)
        K_train, Y_train, rows_train, cols_train = kernels['Train']
        n, nrows, ncols = len(cols_train), len(np.unique(rows_train)), len(np.unique(cols_train))
        print("%d samples: %d rows, %d cols" % (n, nrows, ncols))

        # RLScore
        print("RLScore", end=':')
        tracemalloc.start()
        time_start = time.perf_counter()
        kronrls = CGKronRLS(K1=K_train, K2=K_train, Y=Y_train, label_row_inds=rows_train, label_col_inds=cols_train, regparam=regparam, maxiter=iterations)
        time_train = time.perf_counter()
        P = kronrls.predict(K_train, K_train, rows_train, cols_train)
        time_predict = time.perf_counter()
        print("%.2f seconds" % (time_train-time_start))
        rls_build, rls_train, rls_pred = 0, time_train - time_start, time_predict - time_train
        rls_mem = tracemalloc.get_traced_memory()[1]/10**9
        #rls_mem = 2*K_train.shape[0]*K_train.shape[1]*K_train.itemsize / 10**9
        rls_auc = cindex(Y_train, P)
        tests.append([N, 'RLScore', rls_build, rls_train, rls_pred, rls_mem, rls_auc])
        tracemalloc.stop()

        # Generalized vec-trick
        print("GVT", end=':')
        tracemalloc.start()
        time_start = time.perf_counter()
        mv = mv_kronecker_(K_train, rows_train, cols_train, rows_train, cols_train)
        N = len(Y_train)
        G = LinearOperator((N, N), matvec = lambda v: mv(v) + regparam * v, dtype = np.float64)
        A = minres(G, Y_train, maxiter=iterations, tol=1e-20)[0]
        time_train = time.perf_counter()
        P = mv(A)
        time_predict = time.perf_counter()
        print("%.2f seconds" % (time_train-time_start))
        gvt_build, gvt_train, gvt_pred = 0, time_train - time_start, time_predict - time_train
        gvt_mem = tracemalloc.get_traced_memory()[1]/10**9
        #gvt_mem = 2*K_train.shape[0]*K_train.shape[1]*K_train.itemsize / 10**9
        gvt_auc = cindex(Y_train, P)
        tests.append([N, 'GVT', gvt_build, gvt_train, gvt_pred, gvt_mem, gvt_auc])
        tracemalloc.stop()

        # Explicit Kernel
        print("Kernel", end=':')
        tracemalloc.start()
        time_start = time.perf_counter()
        K_m = kronecker_kernel_(K_train, rows_train, cols_train, rows_train, cols_train)
        time_build = time.perf_counter()
        N = len(Y_train)
        G = LinearOperator((N, N), matvec = lambda v: np.dot(K_m,v) + regparam * v, dtype = np.float64)
        A = minres(G, Y_train, maxiter=iterations, tol=1e-20)[0]
        time_train = time.perf_counter()
        P = np.dot(K_m, A)
        time_predict = time.perf_counter()
        print("%.2f seconds" % (time_train-time_start))
        kkm_build, kkm_train, kkm_pred = time_build - time_start, time_train - time_start, time_predict - time_train
        kkm_mem = tracemalloc.get_traced_memory()[1]/10**9
        #kkm_mem = K_m.shape[0]*K_m.shape[1]*K_m.itemsize / 10**9
        kkm_auc = cindex(Y_train, P)
        tests.append([N, 'Kernel', kkm_build, kkm_train, kkm_pred, kkm_mem, kkm_auc])
        tracemalloc.stop()

        print()
    print()

    tests = pd.DataFrame(tests, columns=['N', 'Method', 'Build', 'Train', 'Predict', 'MEM', 'AUC'])
    tests.to_csv('results/%s' % fn, sep=';', index=False)

    return(tests)

def dense_load(fn='cichonska_MLPK_dense.csv'):

    tests = pd.read_csv('results/%s' % fn, sep=';')

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
    tests.pivot(index='N', columns='Method', values='Train').plot(ax=ax1).set_title('Training time')
    tests.pivot(index='N', columns='Method', values='Predict').plot(ax=ax2).set_title('Prediction time')
    tests.pivot(index='N', columns='Method', values='MEM').plot(ax=ax3).set_title('Kernel MEM')
    plt.show()

if __name__ == '__main__':
    #compare_save_old()
    compare_save()
    compare_load()
    compare_plot()
    #dense_save()
    #dense_load()

