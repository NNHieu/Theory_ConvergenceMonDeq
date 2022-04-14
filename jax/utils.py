import numpy as np
from scipy.sparse import csr_matrix

def vec(A):
    m, n = A.shape[0], A.shape[1]
    return A.reshape(m*n, order='F')

def commutation_matrix_sp(m,n):
    row  = np.arange(m*n)
    col  = row.reshape((m, n), order='F').ravel()
    data = np.ones(m*n, dtype=np.int8)
    K = csr_matrix((data, (row, col)), shape=(m*n, m*n))
    return K