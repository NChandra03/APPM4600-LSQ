# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:04:38 2024

@author: Zara Chandra
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as la
import math
from scipy.integrate import quad

def driver():
    plt.close('all')
    # function you want to approximate
    #f = lambda x: math.exp(x)
    f = lambda x: x * np.exp(-x ** 2)
    # Interval of interest
    a = -3
    b = 3
    # order of approximation
    n = 2
    # Number of points you want to sample in [a,b]
    N = 2
    xeval = np.linspace(a,b,N+1)
    fex = f(xeval)
    M = create_M(xeval, n)
    Q, R = householder_qr(M)
    Qt = np.transpose(Q)
    y_prime = Qt @ fex
    print(R)
    print(y_prime)
    c = back_substitution(R, y_prime)
    plt.scatter(xeval,fex)

def householder_qr(M, tol=1e-10):
    """
    Perform QR decomposition using Householder reflections, and suppress small values in R below the diagonal.

    Parameters:
    M (ndarray): The matrix to decompose.
    tol (float): Tolerance level for zeroing values (default is 1e-10).

    Returns:
    Q (ndarray): Orthogonal matrix.
    R (ndarray): Upper triangular matrix with values below the diagonal suppressed to zero.
    """
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

    # Suppress values in R that are close to zero (non-upper triangular entries)
    for i in range(1, n):
        for j in range(min(i, m)):
            if abs(R[i, j]) < tol:
                R[i, j] = 0.0

    return Q, R

def back_substitution(R, y_prime):
    """
    Perform back substitution to solve the system R * c = y_prime,
    where R is an upper triangular matrix.

    Parameters:
    R (ndarray): An n x n upper triangular matrix.
    y_prime (ndarray): An n-dimensional vector (right-hand side of the equation).

    Returns:
    ndarray: Solution vector c.
    """
    n = R.shape[0]
    c = np.zeros_like(y_prime, dtype=np.float64)

    for i in range(n - 1, -1, -1):
        c[i] = (y_prime[i] - np.dot(R[i, i+1:], c[i+1:])) / R[i, i]
    
    return c

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
    # Each column corresponds to increasing powers of x, from x^0 up to x^n
    M = np.zeros((N, n + 1))
    
    for i in range(n + 1):
        M[:, i] = x_values ** i  # x^i for each column
    
    return M

driver()