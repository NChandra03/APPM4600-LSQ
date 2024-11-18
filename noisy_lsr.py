"""
Now, write an algorithm to solve the least squares regression problem using (1) a
quadratic function, (2) a cubic polynomial, (4) a degree k polynomial, for a few k > 3.
How does the quality of the approximation change as (1) The number of observations
n or polynomial degree k increase? How does the quality change as the ”noise level”
increases”? Perform this experiment with two functions: a polynomial of degree 5 and
f(x) = xe−x
2
.
"""

import numpy as np
import matplotlib.pyplot as plt
import random

from noise_func import uniform
from noise_func import gaussian
from noise_func import generate_noisy_data
from solver_noise import create_M
from solver_noise import householder_qr
from solver_noise import back_substitution


def driver():
  
  N = 100
  a = 0
  b = 10
  n = 2

  # first function
  f1 = lambda x: x**5 - 3x**3 + 5x
  # second function
  f2 = lambda x: x*np.exp(-x**2)
  
  # Number of points to sample in [a, b]
  xeval = np.linspace(a, b, N + 1)
  fex1 = f1(xeval)
  fex2 = f2(xeval)
  noise = normal(N + 1,0.05)
  fex_noise1 = fex1 + noise
  fex_noise2 = fex2 + noise
  
  # Create design matrix M
  M = create_M(xeval,n)
  
  # Perform QR decomposition
  Q, R = householder_qr(M)
  Qt = np.transpose(Q)
  
  # Project f(x) onto the column space of Q and solve for coefficients
  y_prime1 = Qt @ fex1
  y_prime2 = Qt @ fex2

  c1 = back_substitution(R, y_prime1)
  c2 = back_substitution(R, y_prime2)

  
  # Project f(x) onto the column space of Q and solve for coefficients
  y_prime_noise1 = Qt @ fex_noise1
  y_prime_noise2 = Qt @ fex_noise2

  c_noise1 = back_substitution(R, y_prime_noise1)
  c_noise2 = back_substitution(R, y_prime_noise2)

  
  # Generate polynomial values using the coefficients
  x_poly = np.linspace(a, b, 100)
  y_poly1 = sum(c1[i] * x_poly ** i for i in range(n + 1))
  y_poly_noise1 = sum(c_noise1[i] * x_poly ** i for i in range(n + 1))
  y_poly2 = sum(c2[i] * x_poly ** i for i in range(n + 1))
  y_poly_noise2 = sum(c_noise2[i] * x_poly ** i for i in range(n + 1))
  
  # Plot function 1 and the polynomial approximation
  plt.subplot(1, 2, 1)
  plt.scatter(xeval, fex1, color='blue', label='Original function samples: f(x) = x^5 - 3x^3 + 5x')
  plt.scatter(xeval, fex_noise1, color='green', label='Original function samples with noise')
  plt.plot(x_poly, y_poly1, color='red', label='Polynomial approximation')
  plt.plot(x_poly, y_poly_noise1, color='orange', label='Polynomial approximation with noise')
  plt.legend()
  plt.xlabel("x")
  plt.ylabel("f(x) / Polynomial Approximation")
  plt.title("Polynomial Approximation of Function f(x)=x^5-3x^3+5x")

  # Plot function 2 and the polynomial approximation
  plt.subplot(1, 2, 2)
  plt.scatter(xeval, fex2, color='blue', label='Original function samples: f(x) = xe^(-x^2)')
  plt.scatter(xeval, fex_noise2, color='green', label='Original function samples with noise')
  plt.plot(x_poly, y_poly2, color='red', label='Polynomial approximation')
  plt.plot(x_poly, y_poly_noise2, color='orange', label='Polynomial approximation with noise')
  plt.legend()
  plt.xlabel("x")
  plt.ylabel("f(x) / Polynomial Approximation")
  plt.title("Polynomial Approximation of Function f(x)=xe^(-x^2)")

  plt.show()
  plt.savefig("comparison")


driver()
