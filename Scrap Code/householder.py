# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:50:57 2024

@author: Zara Chandra
"""

import numpy as np

def householder_qr(M):
    n, m = M.shape
    Q = np.eye(n)  # Initialize Q as an identity matrix of size n
    R = M.copy()   # Copy of M to transform into R

    for k in range(m):
        # Step 1: Extract the column vector for the k-th Householder transformation
        x = R[k:, k]  # Work on the k-th column from the k-th row downwards
        norm_x = np.linalg.norm(x)
        
        # Step 2: Construct the Householder vector u
        u = x.copy()
        u[0] += np.sign(x[0]) * norm_x  # Adjust the first element
        u = u / np.linalg.norm(u)  # Normalize u
        
        # Step 3: Construct the Householder matrix Hk = I - 2 * u * u.T
        H_k = np.eye(n - k) - 2 * np.outer(u, u)
        
        # Embed H_k into the larger identity matrix for full matrix size
        H = np.eye(n)
        H[k:, k:] = H_k  # Place H_k into the k-th submatrix
        
        # Step 4: Apply H to R and accumulate Q
        R = H @ R  # Apply the transformation to R
        Q = Q @ H.T  # Accumulate the transformation to Q

    return Q, R

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
n = 5
# Number of points to sample in [a, b]
N = 20
a = -3
b = 3
xeval = np.linspace(a, b, N + 1)
M = create_M(xeval,n)
Q, R = householder_qr(M)

print("Matrix Q:\n", Q)
print("Matrix R:\n", R)
