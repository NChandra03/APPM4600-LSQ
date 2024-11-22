# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 15:58:45 2024

@author: Zara Chandra
"""
import numpy as np
import matplotlib.pyplot as plt

def create_M(x_values, n):
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

    R = R[:m, :]
    R = np.triu(R[:m, :])
    return Q[:, :m], R

def back_substitution(R, y_prime, tol=1e-10):
    n = R.shape[0]
    c = np.zeros_like(y_prime, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) > tol:
            c[i] = (y_prime[i] - np.dot(R[i, i+1:], c[i+1:])) / R[i, i]
        else:
            c[i] = 0
    return c

def normal(n, sigma):
    mu = 0
    return np.random.normal(mu, sigma, size=n)

def driver():
    plt.close('all')
    # Function to approximate
    f = lambda x: x ** 5 - 3 * x ** 3 + 5 * x
    # Interval of interest
    a, b = -2, 2
    # Order of approximation
    n = 4
    # Number of points to sample in [a, b]
    N = 30
    xeval = np.linspace(a, b, N + 1)
    fex = f(xeval)

    # Create design matrix M
    M = create_M(xeval, n)

    # Perform QR decomposition
    Q, R = householder_qr(M)
    Qt = np.transpose(Q)

    # Iterate over different sigma values to calculate averaged MSE
    sigma_values = np.linspace(0, 10, 50)
    mse_values = []

    for sigma in sigma_values:
        mse_accumulated = 0
        num_trials = 100  # Number of repetitions for averaging
        for _ in range(num_trials):
            noise = normal(N + 1, sigma)
            fex_noise = fex + noise
            y_prime_noise = Qt @ fex_noise
            c_noise = back_substitution(R, y_prime_noise)
            fex_noise_approx = sum(c_noise[i] * xeval ** i for i in range(n + 1))
            mse_accumulated += np.mean((fex - fex_noise_approx) ** 2)
        mse_average = mse_accumulated / num_trials
        mse_values.append(mse_average)

    # Plot MSE vs sigma
    plt.plot(sigma_values, mse_values, color='blue', label='Averaged MSE vs Sigma')
    plt.xlabel('Sigma (Noise Standard Deviation)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('Averaged MSE vs Sigma for Polynomial Approximation')
    plt.legend()
    plt.grid()
    plt.show()

driver()

