# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 14:55:57 2024

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
    n = 5
    # Number of points you want to sample in [a,b]
    N = 10
    xeval = np.linspace(a,b,N+1)
    fex = f(xeval)
    
    
    plt.scatter(xeval,fex)

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

driver()