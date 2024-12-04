# -*- coding: utf-8 -*-
"""
Created on Mon Dec  2 23:58:28 2024

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

def normal(n, sigma, y):
    y = abs(y)
    mu = 0
    error = []
    for i in range(n):
        # Scale sigma with sqrt(y[i]) to make variance proportional to y[i]
        scaled_sigma = sigma * np.sqrt(y[i])
        error.append(np.random.normal(mu, scaled_sigma))
    return error, scaled_sigma

def weighted_least_squares(M, y, weights):
    """
    Perform Weighted Least Squares (WLS) regression.
    
    Parameters:
    M (ndarray): Design matrix.
    y (ndarray): Observed values (target vector).
    weights (ndarray): Weight vector.
    
    Returns:
    ndarray: Coefficients of the weighted least squares fit.
    """
    # Create diagonal weight matrix
    W = np.diag(weights)
    # Weighted normal equations
    MW = np.sqrt(W) @ M
    yW = np.sqrt(W) @ y
    # QR decomposition of weighted matrix
    Q, R = householder_qr(MW)
    # Solve for coefficients using back substitution
    c = back_substitution(R, Q.T @ yW)
    return c

def driver():
    plt.close('all')
    # Function to approximate
    #f = lambda x: x ** 5 - 2 * x ** 4 + 6 * x ** 3 + 9 * x ** 2 - 7 * x + 64
    f = lambda x: (x - 5) * (x + 5) * (x + 3) * x ** 2
    #f = lambda x: x ** 5 - 3 * x ** 3 + 5 * x
    # Interval of interest
    a, b = -6, 6
    # Order of approximation
    n = 5
    # Number of points to sample in [a, b]
    N = 30
    xeval = np.linspace(a, b, N + 1)
    fex = f(xeval)
    #noise, scaled_sigma = normal(N + 1, 0.2, fex)
    noise, scaled_sigma = normal(N + 1, 10, fex)
    #noise, scaled_sigma = normal(N + 1, 100, fex)
    fex_noise = fex + noise
    
    # Create design matrix M
    M = create_M(xeval, n)
    
    # Ordinary Least Squares (OLS) Solution
    Q, R = householder_qr(M)
    Qt = np.transpose(Q)
    y_prime = Qt @ fex  # Projection for noiseless coefficients
    c_noiseless = back_substitution(R, y_prime)
    
    y_prime_noise = Qt @ fex_noise  # Projection for noisy coefficients
    c_ols = back_substitution(R, y_prime_noise)
    
    # Weighted Least Squares (WLS) Solution
    weights = 1 / np.maximum(np.abs(fex), 1e-10)  # Avoid division by zero
    #log_weights = np.log(weights)
    #normalized_weights = (log_weights - np.min(log_weights)) / (np.max(log_weights) - np.min(log_weights))
    #print(weights)
    plt.figure(1)
    plt.plot(weights)
    plt.yscale('log')
    #plt.ax
    print(weights)
    c_wls = weighted_least_squares(M, fex_noise, weights)
    
    # Generate polynomial values using the coefficients
    x_poly = np.linspace(a, b, 100)
    y_poly_noiseless = sum(c_noiseless[i] * x_poly ** i for i in range(n + 1))
    y_poly_ols = sum(c_ols[i] * x_poly ** i for i in range(n + 1))
    y_poly_wls = sum(c_wls[i] * x_poly ** i for i in range(n + 1))
    
    # Calculate Mean Squared Error (MSE)
    fex_approx_noiseless = sum(c_noiseless[i] * xeval ** i for i in range(n + 1))
    fex_approx_ols = sum(c_ols[i] * xeval ** i for i in range(n + 1))
    fex_approx_wls = sum(c_wls[i] * xeval ** i for i in range(n + 1))
    mse_noiseless = np.mean((fex - fex_approx_noiseless) ** 2)
    mse_ols = np.mean((fex - fex_approx_ols) ** 2)
    mse_wls = np.mean((fex - fex_approx_wls) ** 2)
    
    plt.figure(2)
    # Plot original function, noisy data, OLS, WLS, and noiseless approximation
    plt.scatter(xeval, fex, color='blue', label='Original function samples')
    plt.scatter(xeval, fex_noise, color='green', label='Noisy samples')
    plt.plot(x_poly, y_poly_noiseless, color='orange', label='Noiseless Polynomial Approximation')
    plt.plot(x_poly, y_poly_ols, color='red', label='OLS Polynomial Approximation')
    plt.plot(x_poly, y_poly_wls, color='purple', label='WLS Polynomial Approximation')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x) / Polynomial Approximation")
    plt.title(f"Polynomial Approximations\nMSE (Noiseless): {mse_noiseless:.5f}, MSE (OLS): {mse_ols:.5f}, MSE (WLS): {mse_wls:.5f}")
    plt.show()

driver()




