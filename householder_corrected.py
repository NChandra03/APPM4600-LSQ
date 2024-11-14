# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:40:44 2024

@author: Zara Chandra
"""

import numpy as np
from scipy.linalg import qr


def householder_qr(M, tol=1e-10):
    n, m = M.shape
    Q = np.eye(n)  # Initialize Q as an identity matrix of size n
    R = M.copy()   # Copy of M to transform into R

    for k in range(m):
        # Extract the k-th column vector below the diagonal
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        
        # Construct the Householder vector u
        u = x.copy()
        u[0] += np.sign(x[0]) * norm_x
        u = u / np.linalg.norm(u)
        
        # Construct the Householder transformation matrix H_k
        H_k = np.eye(n - k) - 2 * np.outer(u, u)
        
        # Embed H_k into the larger identity matrix H
        H = np.eye(n)
        H[k:, k:] = H_k
        
        # Apply the transformation to R and accumulate it in Q
        R = H @ R
        Q = Q @ H.T

    # Truncate R to the shape of the economic version
    R = R[:m, :]

    # Set very small values in R to zero for numerical stability
    #R[np.abs(R) < tol] = 0.0
    
    # Zero out all entries below the diagonal in R
    R = np.triu(R[:m, :])

    return Q[:, :m], R

# Example usage:
#M = np.array([[4, 1], [3, 2], [0, 5]])
def create_M(x_values, n):
    """
    Construct the design matrix M for polynomial fitting.
    
    Parameters:
    x_values (ndarray): Array of x-values (of length N) for the data points.
    n (int): The order of the polynomial (degree).
    
    Returns:
    ndarray: The N x (n + 1) design matrix M.
    """
    N = len(x_values)
    M = np.zeros((N, n + 1))
    for i in range(n + 1):
        M[:, i] = x_values ** i  # x^i for each column
    return M

# Order of approximation
n = 2
# Number of points to sample in [a, b]
N = 4
a = -3
b = 3
xeval = np.linspace(a, b, N + 1)
M = create_M(xeval,n)

Q1, R1 = householder_qr(M)

print("Matrix Q:\n", Q1)
print("Matrix R:\n", R1)
print(R1.shape)

Q2, R2 = qr(M, mode='economic')

print("Matrix Q:\n", Q2)
print("Matrix R:\n", R2)
print(R2.shape)