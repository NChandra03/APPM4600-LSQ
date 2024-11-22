# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 18:31:15 2024

@author: Zara Chandra
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import qr

def least_squares_approximation(x, f_values, n):
    # Build the design matrix with polynomial terms 1, x, x^2, ..., x^n
    A = np.vander(x, n + 1, increasing=True)
    
    # QR decomposition of the matrix A
    Q, R = qr(A, mode='economic')
    print(Q)
    print(R)
    
    # Solve for the coefficients in the least squares sense
    coeffs = np.linalg.solve(R, Q.T @ f_values)
    return coeffs

def evaluate_polynomial(coeffs, x):
    # Evaluate the polynomial with the given coefficients at points x
    return np.polyval(coeffs[::-1], x)  # Reverse coeffs for np.polyval order

def driver():
    plt.close('all')
    
    # Define the function to approximate
    f = lambda x: x * np.exp(-x ** 2)
    
    # Interval of interest
    a, b = -3, 3
    
    # Order of approximation
    n = 3
    
    # Number of points to sample in [a, b]
    N = 10
    xeval = np.linspace(a, b, N + 1)
    fex = f(xeval)
    
    # Perform least squares approximation
    coeffs = least_squares_approximation(xeval, fex, n)
    
    # Evaluate the polynomial approximation on a dense grid for smooth plotting
    x_dense = np.linspace(a, b, 200)
    f_approx = evaluate_polynomial(coeffs, x_dense)
    
    # Plot the original function and the polynomial approximation
    plt.plot(x_dense, f(x_dense), label='Original function $f(x) = x e^{-x^2}$')
    plt.plot(x_dense, f_approx, '--', label=f'Least Squares Approximation (order {n})')
    plt.scatter(xeval, fex, color='red', label='Sample points')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Least Squares Polynomial Approximation using QR Decomposition')
    plt.show()

# Run the driver function
driver()
