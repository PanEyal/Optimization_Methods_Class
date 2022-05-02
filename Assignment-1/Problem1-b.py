import numpy as np

def gram_schmidt_QR(A):
    m, n = A.shape
    R = np.zeros((n,n))
    Q = np.zeros((m,n))

    R[0,0] = np.linalg.norm(A[:,0], ord=2)
    Q[:,0] = A[:,0]/R[0,0]

    for i in range(1,n):
        Q[:,i] = A[:,i]
        for j in range(0,i):
            R[j,i] = Q[:,j] @ A[:,i]
            Q[:,i] = Q[:,i] - R[j,i] * Q[:,j]
        R[i,i] = np.linalg.norm(Q[:,i], ord=2)
        Q[:,i] = Q[:,i] / R[i,i]
    
    return Q,R

def gram_schmidt_QR(A):
    m, n = A.shape
    R = np.zeros((n,n))
    Q = np.zeros((m,n))

    R[0,0] = np.linalg.norm(A[:,0], ord=2)
    Q[:,0] = A[:,0]/R[0,0]

    for i in range(1,n):
        Q[:,i] = A[:,i]
        for j in range(0,i):
            R[j,i] = Q[:,j] @ Q[:,i]
            Q[:,i] = Q[:,i] - R[j,i] * Q[:,j]
        R[i,i] = np.linalg.norm(Q[:,i], ord=2)
        Q[:,i] = Q[:,i] / R[i,i]
    
    return Q,R

