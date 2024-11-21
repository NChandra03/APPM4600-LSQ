# -*- coding: utf-8 -*-
"""
Created on Mon Nov 11 15:14:33 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt

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

def householder_qr(M, tol=1e-10):
    n, m = M.shape
    Q = np.eye(n)
    R = M.copy()
    for k in range(m):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        u = x.copy()
        u[0] += np.sign(x[0]) * norm_x
        u = u / np.linalg.norm(u)
        H_k = np.eye(n - k) - 2 * np.outer(u, u)
        H = np.eye(n)
        H[k:, k:] = H_k
        R = H @ R
        Q = Q @ H.T
    for i in range(1, n):
        for j in range(min(i, m)):
            if abs(R[i, j]) < tol:
                R[i, j] = 0.0
    return Q, R

def back_substitution(R, y_prime, tol=1e-10):
    n = R.shape[0]
    c = np.zeros_like(y_prime, dtype=np.float64)
    print(R)
    print(y_prime)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) > tol:  # Only proceed if R[i, i] is not effectively zero
            c[i] = (y_prime[i] - np.dot(R[i, i+1:], c[i+1:])) / R[i, i]
        else:
            # If R[i, i] is zero, set c[i] to 0 or handle as desired
            c[i] = 0  # This might not be unique; adjust as needed for your problem
    return(c)

def driver():
    plt.close('all')
    # Function to approximate
    f = lambda x: x * np.exp(-x ** 2)
    # Interval of interest
    a, b = -3, 3
    # Order of approximation
    n = 5
    # Number of points to sample in [a, b]
    N = 20
    xeval = np.linspace(a, b, N + 1)
    fex = f(xeval)
    
    # Create design matrix M
    M = create_M(xeval, n)
    
    # Perform QR decomposition
    Q, R = householder_qr(M)
    Qt = np.transpose(Q)
    
    # Project f(x) onto the column space of Q and solve for coefficients
    y_prime = Qt @ fex
    c = back_substitution(R, y_prime)
    
    # Generate polynomial values using the coefficients
    x_poly = np.linspace(a, b, 100)
    y_poly = sum(c[i] * x_poly ** i for i in range(n + 1))
    
    # Plot original function values and the polynomial approximation
    plt.scatter(xeval, fex, color='blue', label='Original function samples')
    plt.plot(x_poly, y_poly, color='red', label='Polynomial approximation')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x) / Polynomial Approximation")
    plt.title("Polynomial Approximation of Function")
    plt.show()

driver()
