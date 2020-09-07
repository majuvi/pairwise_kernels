import numpy as np
import random
from sklearn.model_selection import ShuffleSplit, KFold
from math import sqrt

#TODO: heterogeneous setting_indices
#TODO: heterogeneous setting_kernels


# Split row_inds and col_inds into a training set with N samples using ~50% density and
# tests sets with left over samples in Setting 1/2/3/4
#                     cols
#        Train / Test 1 |    Test 3
# rows -------------------------------
#            Test 2     |    Test 4
#
def setting_indices(row_inds, col_inds, N):
    R, C = len(row_inds), len(col_inds)
    rows, cols = np.unique(row_inds), np.unique(col_inds)
    assert R == C
    assert len(rows) == len(cols)

    # Divide indices into setting1, setting2, setting3, setting4
    cols_n = np.ceil(np.sqrt(2*N)).astype(int)
    cols_take = np.random.choice(cols, cols_n, replace=False)
    setting1 = np.isin(row_inds, cols_take) & np.isin(col_inds, cols_take)
    setting2 = np.isin(row_inds, cols_take) & ~np.isin(col_inds, cols_take)
    setting3 = ~np.isin(row_inds, cols_take) & np.isin(col_inds, cols_take)
    setting4 = ~np.isin(row_inds, cols_take) & ~np.isin(col_inds, cols_take)

    # Split setting1 into train/test1 and setting1, setting2, setting3 into test2, test3, test4
    indices = np.arange(C)
    base = indices[setting1]
    ind = np.random.permutation(len(base))
    train = base[ind[:N]]
    test1 = base[ind[N:]]
    test2 = indices[setting2]
    test3 = indices[setting3]
    test4 = indices[setting4]

    return train, test1, test2, test3, test4


# Transform K into a dense matrix, such that column and row indices are surjective
def K_to_dense(K, row_inds, col_inds):
    rows, rows_inverse = np.unique(row_inds, return_inverse=True)
    cols, cols_inverse = np.unique(col_inds, return_inverse=True)
    K = np.array(K[np.ix_(rows, cols)])
    row_indices = np.arange(len(rows))
    col_indices = np.arange(len(cols))
    row_inds = np.array(row_indices[rows_inverse])
    col_inds = np.array(col_indices[cols_inverse])
    return K, row_inds, col_inds

# Transform X into a dense matrix, such row indices are surjective
def X_to_dense(X, row_inds):
    rows, rows_inverse = np.unique(row_inds, return_inverse=True)
    X = np.array(X[rows,])
    row_indices = np.arange(len(rows))
    row_inds = np.array(row_indices[rows_inverse])
    return X, row_inds

# Split given kernel K into train and test kernel matrices in settings 1/2/3/4, where each setting
# has a training kernel K_train (n_train x n_train) and a test kernel K_test (n_test x n_train)
def setting_kernels(K, Y, row_inds, col_inds, N):
    train, test1, test2, test3, test4 = setting_indices(row_inds, col_inds, N)

    kernels = {}

    Y_train = Y[train]
    rows_train, cols_train = row_inds[train], col_inds[train]
    inds_train = np.concatenate([rows_train, cols_train])
    K_train, rows, cols = K_to_dense(K, inds_train, inds_train)
    rows_train, cols_train = rows[:len(rows_train)], rows[len(cols_train):]
    kernels['Train'] = (K_train, Y_train, rows_train, cols_train)

    for setting, test in zip(['Setting 1', 'Setting 2', 'Setting 3', 'Setting 4'], [test1, test2, test3, test4]):
        Y_test = Y[test]
        rows_test, cols_test = row_inds[test], col_inds[test]
        inds_test = np.concatenate([rows_test, cols_test])
        K_test, rows, cols = K_to_dense(K, inds_test, inds_train)
        rows_test, cols_test = rows[:len(rows_test)], rows[len(cols_test):]
        kernels[setting] = (K_test, Y_test, rows_test, cols_test)

    return(kernels)



# Split given kernel K into train and test kernel matrices in settings 1/2/3/4, where each setting
# has a training kernel K_train (n_train x n_train) and a test kernel K_test (n_test x n_train)
def setting_kernels_homogeneous(K, Y, row_inds, col_inds, split_ratio=0.25, setting='Setting 1'):

    if setting == 'Setting 1':
        train, test = setting1_split(row_inds, col_inds, split_ratio)
    elif setting == 'Setting 2':
        train, test = setting2_split(row_inds, col_inds, split_ratio)
    elif setting == 'Setting 3':
        train, test = setting3_split(row_inds, col_inds, split_ratio)
    elif setting == 'Setting 4':
        train, test = setting4_split(row_inds, col_inds, split_ratio)

    kernels = {}

    Y_train = Y[train]
    rows_train, cols_train = row_inds[train], col_inds[train]
    inds_train = np.concatenate([rows_train, cols_train])
    K_train, rows, cols = K_to_dense(K, inds_train, inds_train)
    rows_train, cols_train = rows[:len(rows_train)], rows[len(cols_train):]
    kernels['Train'] = (K_train, Y_train, rows_train, cols_train)

    Y_test = Y[test]
    rows_test, cols_test = row_inds[test], col_inds[test]
    inds_test = np.concatenate([rows_test, cols_test])
    K_test, rows, cols = K_to_dense(K, inds_test, inds_train)
    rows_test, cols_test = rows[:len(rows_test)], rows[len(cols_test):]
    kernels['Test'] = (K_test, Y_test, rows_test, cols_test)

    return(kernels)


# Split given data matrices XD/XT into train and test data matrices in settings 1/2/3/4
def setting_kernels_heterogeneous(KD, KT, Y, row_inds, col_inds, split_ratio=0.25, setting='Setting 1'):

    if setting == 'Setting 1':
        train, test = setting1_split(row_inds, col_inds, split_ratio)
    elif setting == 'Setting 2':
        train, test = setting2_split(row_inds, col_inds, split_ratio)
    elif setting == 'Setting 3':
        train, test = setting3_split(row_inds, col_inds, split_ratio)
    elif setting == 'Setting 4':
        train, test = setting4_split(row_inds, col_inds, split_ratio)

    data = {}

    Y_train = Y[train]
    rows_train, cols_train = row_inds[train], col_inds[train]
    KD_train, rows, rows = K_to_dense(KD, rows_train, rows_train)
    KT_train, cols, cols = K_to_dense(KT, cols_train, cols_train)
    data['Train'] = (KD_train, KT_train, Y_train, rows, cols)

    Y_test = Y[test]
    rows_test, cols_test = row_inds[test], col_inds[test]
    KD_test, rows1, rows2 = K_to_dense(KD, rows_test, rows_train)
    KT_test, cols1, cols2 = K_to_dense(KT, cols_test, cols_train)
    data['Test'] = (KD_test, KT_test, Y_test, rows1, cols1)

    return(data)


# Split given data matrices XD/XT into train and test data matrices in settings 1/2/3/4
def setting_kernels_heterogeneous_CV(KD, KT, Y, row_inds, col_inds, cv=None, setting='Setting 1'):

    cv = cv if not cv is None else (3 if setting == 'Setting 4' else 9)
    cv_f = {'Setting 1': setting1_cv, 'Setting 2': setting2_cv, 'Setting 3': setting3_cv, 'Setting 4': setting4_cv}[setting]

    for train, test in cv_f(row_inds, col_inds, cv):

        Y_train = Y[train]
        rows_train, cols_train = row_inds[train], col_inds[train]
        KD_train, rows, rows = K_to_dense(KD, rows_train, rows_train)
        KT_train, cols, cols = K_to_dense(KT, cols_train, cols_train)
        train = (KD_train, KT_train, Y_train, rows, cols)

        Y_test = Y[test]
        rows_test, cols_test = row_inds[test], col_inds[test]
        KD_test, rows1, rows2 = K_to_dense(KD, rows_test, rows_train)
        KT_test, cols1, cols2 = K_to_dense(KT, cols_test, cols_train)
        test = (KD_test, KT_test, Y_test, rows1, cols1)

        yield(train, test)


def setting1_split(row_inds, col_inds, split=0.25):
    assert len(row_inds) == len(col_inds)
    N = len(row_inds)
    cv = ShuffleSplit(n_splits=1, test_size=split, random_state=0)
    train, test1 = next(cv.split(row_inds))
    return train, test1

def setting2_split(row_inds, col_inds, split=0.25):
    assert len(row_inds) == len(col_inds)
    N = len(row_inds)
    rows, cols = np.unique(row_inds), np.unique(col_inds)
    cv = ShuffleSplit(n_splits=1, test_size=split, random_state=0)
    train_cols, test_cols = next(cv.split(cols))
    train_cols, test_cols = cols[train_cols], cols[test_cols]
    indices = np.arange(N)
    train = indices[np.isin(col_inds, train_cols)]
    test2 = indices[np.isin(col_inds, test_cols)]
    return train, test2

def setting3_split(row_inds, col_inds, split=0.25):
    assert len(row_inds) == len(col_inds)
    N = len(row_inds)
    rows, cols = np.unique(row_inds), np.unique(col_inds)
    cv = ShuffleSplit(n_splits=1, test_size=split, random_state=0)
    train_rows, test_rows = next(cv.split(rows))
    train_rows, test_rows = rows[train_rows], rows[test_rows]
    indices = np.arange(N)
    train = indices[np.isin(row_inds, train_rows)]
    test3 = indices[np.isin(row_inds, test_rows)]
    return train, test3

def setting4_split(row_inds, col_inds, split=0.5*(-1+sqrt(3))):
    assert len(row_inds) == len(col_inds)
    N = len(row_inds)
    rows, cols = np.unique(row_inds), np.unique(col_inds)
    cv1 = ShuffleSplit(n_splits=1, test_size=split, random_state=0)
    cv2 = ShuffleSplit(n_splits=1, test_size=split, random_state=0)
    train_cols, test_cols = next(cv1.split(cols))
    train_cols, test_cols = cols[train_cols], cols[test_cols]
    train_rows, test_rows = next(cv2.split(rows))
    train_rows, test_rows = rows[train_rows], rows[test_rows]
    indices = np.arange(N)
    train = indices[np.isin(row_inds, train_rows) & np.isin(col_inds, train_cols)]
    test4 = indices[np.isin(row_inds, test_rows) & np.isin(col_inds, test_cols)]
    return train, test4

def setting1_cv(row_inds, col_inds, cv=9):
    assert len(row_inds) == len(col_inds)
    cv = KFold(n_splits=cv, random_state=0)
    indexes = [(train_index, test_index) for train_index, test_index in cv.split(row_inds)]
    return indexes

def setting2_cv(row_inds, col_inds, cv=9):
    assert len(row_inds) == len(col_inds)
    N = len(row_inds)
    rows, cols = np.unique(row_inds), np.unique(col_inds)
    cv = KFold(n_splits=cv, random_state=0)
    indices = np.arange(N)
    indexes = []
    for train_cols, test_cols in cv.split(cols):
        train_cols, test_cols = cols[train_cols], cols[test_cols]
        train = indices[np.isin(col_inds, train_cols)]
        test2 = indices[np.isin(col_inds, test_cols)]
        indexes.append((train, test2))
    return indexes

def setting3_cv(row_inds, col_inds, cv=9):
    assert len(row_inds) == len(col_inds)
    N = len(row_inds)
    rows, cols = np.unique(row_inds), np.unique(col_inds)
    cv = KFold(n_splits=cv, random_state=0)
    indices = np.arange(N)
    indexes = []
    for train_rows, test_rows in cv.split(rows):
        train_rows, test_rows = rows[train_rows], rows[test_rows]
        train = indices[np.isin(row_inds, train_rows)]
        test3 = indices[np.isin(row_inds, test_rows)]
        indexes.append((train, test3))
    return indexes

def setting4_cv(row_inds, col_inds, cv=3):
    assert len(row_inds) == len(col_inds)
    N = len(row_inds)
    rows, cols = np.unique(row_inds), np.unique(col_inds)
    cv1 = KFold(n_splits=cv, random_state=0)
    cv2 = KFold(n_splits=cv, random_state=0)
    indices = np.arange(N)
    indexes = []
    for train_cols, test_cols in cv1.split(cols):
        for train_rows, test_rows in cv2.split(rows):
            train_cols, test_cols = cols[train_cols], cols[test_cols]
            train_rows, test_rows = rows[train_rows], rows[test_rows]
            train = indices[np.isin(row_inds, train_rows) & np.isin(col_inds, train_cols)]
            test4 = indices[np.isin(row_inds, test_rows) & np.isin(col_inds, test_cols)]
            indexes.append((train, test4))
    return indexes


if __name__ == '__main__':
    from data import load_metz

    XD, XT, Y, drug_inds, target_inds = load_metz()
    print("Labeled pairs %d" % len(Y))
    print("Total rows %d, cols %d" % (len(np.unique(drug_inds)), len(np.unique(target_inds))))

    print("=== Setting 1 ===")
    train_index, test_index = setting1_split(drug_inds, target_inds)
    print(train_index)
    print(test_index)
    n_train, n_test, n = len(train_index), len(test_index), len(train_index) + len(test_index)
    print("Train Labeled pairs %d, Test labelled pairs %d (%.1f%%)" % (n_train, n_test, 100*n_test/float(n)))
    print("Train rows %d, Train cols %d" % (len(np.unique(drug_inds[train_index])), len(np.unique(target_inds[train_index]))))
    print("Test rows %d, Test cols %d" % (len(np.unique(drug_inds[test_index])), len(np.unique(target_inds[test_index]))))

    print("=== Setting 2 ===")
    train_index, test_index = setting2_split(drug_inds, target_inds)
    print(train_index)
    print(test_index)
    n_train, n_test, n = len(train_index), len(test_index), len(train_index) + len(test_index)
    print("Train Labeled pairs %d, Test labelled pairs %d (%.1f%%)" % (n_train, n_test, 100*n_test/float(n)))
    print("Train rows %d, Train cols %d" % (len(np.unique(drug_inds[train_index])), len(np.unique(target_inds[train_index]))))
    print("Test rows %d, Test cols %d" % (len(np.unique(drug_inds[test_index])), len(np.unique(target_inds[test_index]))))

    print("=== Setting 3 ===")
    train_index, test_index = setting3_split(drug_inds, target_inds)
    print(train_index)
    print(test_index)
    n_train, n_test, n = len(train_index), len(test_index), len(train_index) + len(test_index)
    print("Train Labeled pairs %d, Test labelled pairs %d (%.1f%%)" % (n_train, n_test, 100*n_test/float(n)))
    print("Train rows %d, Train cols %d" % (len(np.unique(drug_inds[train_index])), len(np.unique(target_inds[train_index]))))
    print("Test rows %d, Test cols %d" % (len(np.unique(drug_inds[test_index])), len(np.unique(target_inds[test_index]))))

    print("=== Setting 4 ===")
    train_index, test_index = setting4_split(drug_inds, target_inds)
    print(train_index)
    print(test_index)
    n_train, n_test, n = len(train_index), len(test_index), len(train_index) + len(test_index)
    print("Train Labeled pairs %d, Test labelled pairs %d (%.1f%%)" % (n_train, n_test, 100*n_test/float(n)))
    print("Train rows %d, Train cols %d" % (len(np.unique(drug_inds[train_index])), len(np.unique(target_inds[train_index]))))
    print("Test rows %d, Test cols %d" % (len(np.unique(drug_inds[test_index])), len(np.unique(target_inds[test_index]))))

