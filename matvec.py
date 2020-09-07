import numpy as np
from scipy import sparse as sp

from rlscore.utilities.pairwise_kernel_operator import PairwiseKernelOperator
from rlscore.utilities.sampled_kronecker_products import sampled_vec_trick

# Matrix vector products
#     return a linear operator f:R^d->R^d such that y=f(a)
#
# K1=K2  Kernel (N x N)               Kernel ((m x q) x ( m x q))    GVT (N x N)         RLScore (N x N)
# ----------------------------------------------------------------------------------------------------------
#        kronecker_kernel             kronecker_kernel_full          mv_kronecker        pko_kronecker
#        cartesian_kernel             cartesian_kernel_full          mv_cartesian        pko_cartesian
#        linear_kernel                linear_kernel_full             mv_linear           pko_linear
#        poly2d_kernel                poly2d_kernel_full             mv_poly2d           pko_poly2d
#   X    symmetric_kernel             symmetric_kernel_full          mv_symmetric        pko_symmetric
#   X    antisymmetric_kernel         antisymmetric_kernel_full      mv_antisymmetric    pko_antisymmetric
#   X    mlpk_kernel                  mlpk_kernel_full               mv_mlpk             pko_mlpk

def kronecker_kernel(K1, K2, rows1, cols1, rows2, cols2):
    #Returns Kronecker kernel
    assert len(rows1) == len(cols1)
    assert len(rows2) == len(cols2)
    n = len(rows1)
    m = len(rows2)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            k_ij = K1[rows1[i], rows2[j]]
            g_ij = K2[cols1[i], cols2[j]]
            val = k_ij * g_ij
            K[i,j] = val 
    return K

def cartesian_kernel(K1, K2, rows1, cols1, rows2, cols2):
    #Returns Cartesian kernel
    #assumption: K1 and K2 symmetric matrices
    assert len(rows1) == len(cols1)
    assert len(rows2) == len(cols2)
    assert K1.shape[0] == K1.shape[1]
    assert K2.shape[0] == K2.shape[1]
    n = len(rows1)
    m = len(rows2)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            k_ij = K1[rows1[i], rows2[j]]
            g_ij = K2[cols1[i], cols2[j]]
            val = k_ij * (cols1[i] == cols2[j]) + g_ij * (rows1[i] == rows2[j])
            K[i,j] = val 
    return K


def linear_kernel(K1, K2, rows1, cols1, rows2, cols2):
    #Returns Linear kernel
    #assumption: K1 and K2 symmetric matrices
    assert len(rows1) == len(cols1)
    assert len(rows2) == len(cols2)
    n = len(rows1)
    m = len(rows2)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            k_ij = K1[rows1[i], rows2[j]]
            g_ij = K2[cols1[i], cols2[j]]
            val = k_ij + g_ij
            K[i,j] = val
    return K

def poly2d_kernel(K1, K2, rows1, cols1, rows2, cols2):
    #Returns Kronecker kernel
    assert len(rows1) == len(cols1)
    assert len(rows2) == len(cols2)
    n = len(rows1)
    m = len(rows2)
    K = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            k_ij = K1[rows1[i], rows2[j]]
            g_ij = K2[cols1[i], cols2[j]]
            val = k_ij**2 + 2*k_ij*g_ij + g_ij**2
            #val = 1 + 2*k_ij + 2*g_ij + 2*k_ij*g_ij + k_ij**2 + g_ij**2
            K[i,j] = val
    return K

def symmetric_kernel(K, rows1, cols1, rows2, cols2):
    #Returns Cartesian kernel
    #assumption: K1 and K2 symmetric matrices
    assert len(rows1) == len(cols1)
    assert len(rows2) == len(cols2)
    n = len(rows1)
    m = len(rows2)
    K_new = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            val = K[rows1[i], rows2[j]] * K[cols1[i], cols2[j]] + K[rows1[i], cols2[j]] * K[cols1[i], rows2[j]]
            K_new[i,j] = val 
    return K_new

def antisymmetric_kernel(K, rows1, cols1, rows2, cols2):
    #Returns Cartesian kernel
    #assumption: K1 and K2 symmetric matrices
    assert len(rows1) == len(cols1)
    assert len(rows2) == len(cols2)
    n = len(rows1)
    m = len(rows2)
    K_new = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            val = K[rows1[i], rows2[j]] * K[cols1[i], cols2[j]] - K[rows1[i], cols2[j]] * K[cols1[i], rows2[j]]
            K_new[i,j] = val
    return K_new

def mlpk_kernel(K, rows1, cols1, rows2, cols2):
    #Returns Cartesian kernel
    #assumption: K1 and K2 symmetric matrices
    assert len(rows1) == len(cols1)
    assert len(rows2) == len(cols2)
    n = len(rows1)
    m = len(rows2)
    K_new = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            val = (K[rows1[i], rows2[j]] + K[cols1[i], cols2[j]] - K[rows1[i], cols2[j]] - K[cols1[i], rows2[j]])**2.
            K_new[i,j] = val 
    return K_new

def com_mat(size):
    rows = []
    cols = []
    vals = []
    for i in range(size):
        for j in range(size):
            rows.append( i*size + j )
            cols.append( j*size + i )
            vals.append(1)
    P = sp.coo_matrix((vals, (rows, cols)))
    P = P.todense()
    return P

def q_mat(size):
    rows = []
    cols = []
    vals = []
    for i in range(size):
        for j in range(size):
            rows.append( i*size + j )
            cols.append( i*size + i )
            vals.append(1)
    P = sp.coo_matrix((vals, (rows, cols)))
    P = P.todense()
    return P

def kronecker_kernel_full(K1, K2):
    return np.kron(K1, K2)

def cartesian_kernel_full(K1, K2):
    I1 = np.eye(K1.shape[0])
    I2 = np.eye(K2.shape[0])
    return np.kron(K1, I2) + np.kron(I1, K2)

def linear_kernel_full(K1, K2):
    n = K1.shape[0]
    m = K2.shape[0]
    O1 = np.ones((n, n))
    O2 = np.ones((m, m))
    return np.kron(K1, O2) + np.kron(O1, K2)

def poly2d_kernel_full(K1, K2):
    n = K1.shape[0]
    m = K2.shape[0]
    O1 = np.ones((n, n))
    O2 = np.ones((m, m))

    K_2d = np.kron(np.multiply(K1, K1), O2) + 2*np.kron(K1, K2) + np.kron(O1, np.multiply(K2, K2))
    #K_2d = np.eye(m*n) + 2*np.kron(K1, O2) + 2*np.kron(O1, K2)) +  2*np.kron(K1, K2) + np.kron(np.multiply(K1, K1), O2) + np.kron(O1, np.multiply(K2, K2))
    return K_2d

def symmetric_kernel_full(K):
    m = K.shape[0]
    I = np.eye(m**2)
    P = com_mat(m)
    K = np.kron(K, K)
    K = K + np.dot(P, K)
    return K

def antisymmetric_kernel_full(K):
    m = K.shape[0]
    I = np.eye(m**2)
    P = com_mat(m)
    K = np.kron(K, K)
    K = K - np.dot(P, K)
    return K

def mlpk_kernel_full(K):
    m = K.shape[0]
    I = np.eye(m**2)
    P = com_mat(m)
    O = np.ones((m, m))
    Q = q_mat(m)
    I_P = I+P
    K_sq = np.multiply(K, K)
    KoK = np.kron(K, K)
    K_mlpk = np.dot(I_P, (np.kron(K_sq, O) + np.kron(O, K_sq))) + 2*np.dot(I_P, KoK) - 2*(np.dot(I_P, np.dot(Q, KoK) ) + np.dot(np.dot(KoK, Q.T), I_P))
    return K_mlpk

def mv_kronecker(K1, K2, rows1, cols1, rows2, cols2):
    # Kronecker product kernel
    def mv(v):
        return sampled_vec_trick(v, K2, K1, cols1, rows1, cols2, rows2)
    return mv

def mv_cartesian(K1, K2, rows1, cols1, rows2, cols2):
    I1 = np.eye(K1.shape[0])
    I2 = np.eye(K2.shape[0])
    # Cartesian kernel
    def mv(v):
        return sampled_vec_trick(v, I2, K1, cols1, rows1, cols2, rows2) + \
               sampled_vec_trick(v, K2, I1, cols1, rows1, cols2, rows2)
    return mv

def mv_linear(K1, K2, rows1, cols1, rows2, cols2):
    n = K1.shape[0]
    m = K2.shape[0]
    O1 = np.ones((n, n))
    O2 = np.ones((m, m))
    # Cartesian kernel
    def mv(v):
        return sampled_vec_trick(v, O2, K1, cols1, rows1, cols2, rows2) + \
               sampled_vec_trick(v, K2, O1, cols1, rows1, cols2, rows2)
    return mv

def mv_poly2d(K1, K2, rows1, cols1, rows2, cols2):
    n = K1.shape[0]
    m = K2.shape[0]
    O1 = np.ones((n, n))
    O2 = np.ones((m, m))
    K1_sq = np.multiply(K1, K1)
    K2_sq = np.multiply(K2, K2)
    # Polynomial 2nd degree kernel
    def mv(v):
        return sampled_vec_trick(v, O2, K1_sq, cols1, rows1, cols2, rows2) + \
               2*sampled_vec_trick(v, K2, K1, cols1, rows1, cols2, rows2) + \
               sampled_vec_trick(v, K2_sq, O1, cols1, rows1, cols2, rows2)
        #return v + 2*sampled_vec_trick(v, O2, K1, cols, rows, cols, rows) + \
        #       2*sampled_vec_trick(v, K2, O1, cols, rows, cols, rows) + \
        #       sampled_vec_trick(v, O2, K1_sq, cols, rows, cols, rows) + \
        #       2*sampled_vec_trick(v, K2, K1, cols, rows, cols, rows) + \
        #       sampled_vec_trick(v, K2_sq, O1, cols, rows, cols, rows) + regparam * v
    return mv

def mv_symmetric(K, rows1, cols1, rows2, cols2):
    # Symmetric Kronecker product kernel
    def mv(v):
        return sampled_vec_trick(v, K, K, cols1, rows1, cols2, rows2) + \
               sampled_vec_trick(v, K, K, cols1, rows1, rows2, cols2)
    return mv

def mv_antisymmetric(K, rows1, cols1, rows2, cols2):
    # Symmetric Kronecker product kernel
    def mv(v):
        return sampled_vec_trick(v, K, K, cols1, rows1, cols2, rows2) - \
               sampled_vec_trick(v, K, K, cols1, rows1, rows2, cols2)
    return mv

def mv_mlpk(K, rows1, cols1, rows2, cols2):
    # Metric learning pairwise kernel
    m = K.shape[0]
    K_sq = np.multiply(K, K)
    O = np.ones((m, m))
    def mv(v):
        A = sampled_vec_trick(v, K_sq, O, cols1, rows1, cols2, rows2) + sampled_vec_trick(v, O, K_sq, cols1, rows1, cols2, rows2)
        B = sampled_vec_trick(v, K_sq, O, cols1, rows1, rows2, cols2) + sampled_vec_trick(v, O, K_sq, cols1, rows1, rows2, cols2)
        C = 2* (sampled_vec_trick(v, K, K, cols1, rows1, cols2, rows2) + sampled_vec_trick(v, K, K, cols1, rows1, rows2, cols2) )
        D = 2* (sampled_vec_trick(v, K, K, cols1, cols1, cols2, rows2) + sampled_vec_trick(v, K, K, cols1, rows1, rows2, rows2) )
        E = 2* (sampled_vec_trick(v, K, K, rows1, cols1, cols2, cols2) + sampled_vec_trick(v, K, K, rows1, rows1, rows2, cols2) )
        return A + B + C - D - E
    return mv

def pko_kronecker(K1, K2, rows1, cols1, rows2, cols2):
    pko = PairwiseKernelOperator([K1], [K2], [rows1], [cols1], [rows2], [cols2], weights=[1.0])
    return pko

def pko_cartesian(K1, K2, rows1, cols1, rows2, cols2):
    n, d = K1.shape
    m, k = K2.shape
    I1 = np.eye(n, d)
    I2 = np.eye(m, k)
    pko = PairwiseKernelOperator([K1, I1], [I2, K2], [rows1, rows1], [cols1, cols1], [rows2, rows2], [cols2, cols2], weights=[1.0, 1.0])
    return pko

def pko_linear(K1, K2, rows1, cols1, rows2, cols2):
    n, d = K1.shape#[0]
    m, k = K2.shape#[0]
    O1 = np.ones((n, d))
    O2 = np.ones((m, k))
    #print("pko_linear:")
    #print(K1.shape, O2.shape)
    #print(O1.shape, K2.shape)
    #print(rows1.max(), cols1.max(), rows2.max(), cols2.max())
    pko = PairwiseKernelOperator([K1, O1], [O2, K2], [rows1, rows1], [cols1, cols1], [rows2, rows2], [cols2, cols2], weights=[1.0, 1.0])
    return pko

def pko_poly2d(K1, K2, rows1, cols1, rows2, cols2):
    n, d = K1.shape#[0]
    m, k = K2.shape#[0]
    O1 = np.ones((n, d))
    O2 = np.ones((m, k))
    K1_sq = np.multiply(K1, K1)
    K2_sq = np.multiply(K2, K2)
    pko = PairwiseKernelOperator([K1_sq, K1, O1], [O2, K2, K2_sq], [rows1, rows1, rows1], [cols1, cols1, cols1], [rows2, rows2, rows2], [cols2, cols2, cols2], weights=[1.0, 2.0, 1.0])
    return pko

def pko_symmetric(K, rows1, cols1, rows2, cols2):
    pko = PairwiseKernelOperator([K, K], [K, K], [rows1, rows1], [cols1, cols1], [rows2, cols2], [cols2, rows2], weights=[1.0, 1.0])
    return pko

def pko_antisymmetric(K, rows1, cols1, rows2, cols2):
    pko = PairwiseKernelOperator([K, K], [K, K], [rows1, rows1], [cols1, cols1], [rows2, cols2], [cols2, rows2], weights=[1.0, -1.0])
    return pko

def pko_mlpk(K, rows1, cols1, rows2, cols2):
    m, d = K.shape#[0]
    K_sq = np.multiply(K, K)
    O = np.ones((m, d))
    pko = PairwiseKernelOperator([O, K_sq, O, K_sq, K, K, K, K, K, K], [K_sq, O, K_sq, O, K, K, K, K, K, K],
                                 [rows1, rows1, rows1, rows1, rows1, rows1, cols1, rows1, cols1, rows1],
                                 [cols1, cols1, cols1, cols1, cols1, cols1, cols1, cols1, rows1, rows1],
                                 [rows2, rows2, cols2, cols2, rows2, cols2, rows2, rows2, cols2, cols2],
                                 [cols2, cols2, rows2, rows2, cols2, rows2, cols2, rows2, cols2, rows2],
                                 weights=[1.0, 1.0, 1.0, 1.0, 2.0, 2.0, -2.0, -2.0, -2.0, -2.0])
    return pko


if __name__=="__main__":
    A = [[1,2],[3,4]]
    A = np.array(A)
    A= np.dot(A, A.T)
    B = [[0,5],[6,7]]
    A = np.array(A)
    B = np.array(B)
    rows1 = np.array([0,0,1,1])
    cols1 = np.array([0,1,0,1])
    K = symmetric_kernel_full(A)
    print(K)
    K = symmetric_kernel(A, rows1, cols1, rows1, cols1)
    print(K)

