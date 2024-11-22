# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 19:11:37 2024

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

    # Zero out all entries below the diagonal in R
    R = np.triu(R[:m, :])

    return Q[:, :m], R

def back_substitution(R, y_prime, tol=1e-10):
    n = R.shape[0]
    c = np.zeros_like(y_prime, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) > tol:  # Only proceed if R[i, i] is not effectively zero
            c[i] = (y_prime[i] - np.dot(R[i, i+1:], c[i+1:])) / R[i, i]
        else:
            c[i] = 0  # Set c[i] to 0 if R[i, i] is effectively zero
    return c

def normal(n, sigma):
    mu = 0
    return np.random.normal(mu, sigma, size=n)

def driver():
    plt.close('all')
    # Function to approximate
    f = lambda x: x * np.exp(-x ** 2)
    # Interval of interest
    a, b = -3, 3
    # Order of approximation
    n = 5
    # Number of points to sample in [a, b]
    N = 30
    xeval = np.linspace(a, b, N + 1)
    fex = f(xeval)
    noise = normal(N + 1, 0.5)
    fex_noise = fex + noise
    
    # Create design matrix M
    M = create_M(xeval, n)
    
    # Perform QR decomposition
    Q, R = householder_qr(M)
    Qt = np.transpose(Q)
    
    # Project f(x) onto the column space of Q and solve for coefficients
    y_prime = Qt @ fex
    c = back_substitution(R, y_prime)
    
    # Project f(x) onto the column space of Q and solve for coefficients
    y_prime_noise = Qt @ fex_noise
    c_noise = back_substitution(R, y_prime_noise)
    
    # Generate polynomial values using the coefficients
    x_poly = np.linspace(a, b, 100)
    y_poly = sum(c[i] * x_poly ** i for i in range(n + 1))
    y_poly_noise = sum(c_noise[i] * x_poly ** i for i in range(n + 1))
    
    # Calculate Mean Squared Error (MSE)
    fex_approx = sum(c[i] * xeval ** i for i in range(n + 1))
    fex_noise_approx = sum(c_noise[i] * xeval ** i for i in range(n + 1))
    
    mse_noiseless = np.mean((fex - fex_approx) ** 2)
    mse_noisy = np.mean((fex - fex_noise_approx) ** 2)
    
    # Plot original function values and the polynomial approximation
    plt.scatter(xeval, fex, color='blue', label='Original function samples')
    plt.scatter(xeval, fex_noise, color='green', label='Original function samples with noise')
    plt.plot(x_poly, y_poly, color='red', label='Polynomial approximation (Noiseless)')
    plt.plot(x_poly, y_poly_noise, color='orange', label='Polynomial approximation (Noisy)')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x) / Polynomial Approximation")
    plt.title(f"Polynomial Approximation of Function\nMSE (Noiseless): {mse_noiseless:.5f}, MSE (Noisy): {mse_noisy:.5f}")
    plt.show()

driver()
