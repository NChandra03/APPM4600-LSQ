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

from noise_func import uniform, normal, generate_noisy_data
from solver_noise import create_M, householder_qr, back_substitution

def driver():
  # Number of points to sample
  N = 100
  # Interval of interest
  a = 0
  b = 2
  # Polynomial degrees to consider
  degrees = [2, 3, 4, 5]

  # Functions to approximate
  f1 = lambda x: x**5 - 3*x**3 + 5*x 
  f2 = lambda x: x*np.exp(-x**2)

  # Sample points
  xeval = np.linspace(a, b, N + 1)
  noise = normal(N + 1, 0.05)
  xeval_noisy = xeval+noise

  # using gaussian/normal noise distribution to replicate real data
  fex1 = f1(xeval)
  fex2 = f2(xeval)

  #fex_noise_y1 = fex1 + noise
  #fex_noise_y2 = fex2 + noise

  fex_noise1 = f1(xeval_noisy)
  fex_noise2 = f2(xeval_noisy)

  #fex_noise_xy1 = f1(xeval_noisy) + noise
  #fex_noise_xy1 = f2(xeval_noisy) + noise

  fig, axes = plt.subplots(len(degrees), 2, figsize=(12, len(degrees) * 4))

  for i, n in enumerate(degrees):
    # noiseless input design matrix
    M = create_M(xeval, n)
    Q, R = householder_qr(M)
    Qt = np.transpose(Q)
    
    # noisy input design matrix
    Mnoisy = create_M(xeval_noisy, n)
    Qnoisy, Rnoisy = householder_qr(Mnoisy)
    Qtnoisy = np.transpose(Qnoisy)

    # Solve for coefficients (noise-free)
    y_prime1 = Qt @ fex1
    y_prime2 = Qt @ fex2
    c1 = back_substitution(R, y_prime1)
    c2 = back_substitution(R, y_prime2)

    # Solve for coefficients (with input noise)
    y_prime_noise1 = Qtnoisy @ fex_noise1
    y_prime_noise2 = Qtnoisy @ fex_noise2
    c_noise1 = back_substitution(R, y_prime_noise1)
    c_noise2 = back_substitution(R, y_prime_noise2)

    # Generate polynomial values with noise
    x_poly = np.linspace(a, b, 101)
    #x_poly_noise = x_poly + xnoise
    y_poly1 = sum(c1[i] * x_poly ** i for i in range(n + 1))
    y_poly_noise1 = sum(c_noise1[i] * x_poly ** i for i in range(n + 1))
    y_poly2 = sum(c2[i] * x_poly ** i for i in range(n + 1))
    y_poly_noise2 = sum(c_noise2[i] * x_poly ** i for i in range(n + 1))

    # Plot for Function 1
    axes[i, 0].scatter(xeval, fex1, color='black', label=r'Function $f(x) = x^5 - 3x^3 + 5x$')
    axes[i, 0].scatter(xeval, fex_noise1, color='red', label=r'Function with input noise')
    axes[i, 0].plot(x_poly, y_poly1, color='green', label=f'Polynomial approximation (deg {n})')
    axes[i, 0].plot(x_poly, y_poly_noise1, color='orange', label='Polynomial approxmation with input noise')
    #axes[i, 0].set_yscale('log')
    axes[i, 0].legend()
    axes[i, 0].set_xlabel("x")
    axes[i, 0].set_ylabel("f(x) / Polynomial Approximation")
    axes[i, 0].set_title(r"$f(x) = x^5 - 3x^3 + 5x$ with Degree " + f"{n}" + " Polynomial Approximation")

    # Plot for Function 2
    axes[i, 1].scatter(xeval, fex2, color='black', label=r'Function $f(x) = xe^{-x^2}$')
    axes[i, 1].scatter(xeval, fex_noise2, color='red', label=r'Function with input noise')
    axes[i, 1].plot(x_poly, y_poly2, color='green', label=f'Polynomial approximation (deg {n})')
    axes[i, 1].plot(x_poly, y_poly_noise2, color='orange', label=f'Polynomial approximation with input noise')
    axes[i, 1].legend()
    axes[i, 1].set_xlabel("x")
    axes[i, 1].set_ylabel("f(x) / Polynomial Approximation")
    axes[i, 1].set_title(f"Function 2: Degree {n}")
    axes[i, 1].set_title(r"$f(x) = x e^{-x^2}$ with Degree " + f"{n}" + " Polynomial Approximation")

  plt.tight_layout()
  plt.savefig("xnoise_compare_degree.png", dpi=300)
  plt.show()
  plt.clf()
  plt.figure()

  # Choose order for noise level analysis
  n = 5
  noise_levels = [0.05, 0.1, 0.2, 0.3]

  # noiseless input design matrix
  M = create_M(xeval, n)
  Q, R = householder_qr(M)
  Qt = np.transpose(Q)

  # Solve for coefficients (noise-free)
  y_prime1 = Qt @ fex1
  y_prime2 = Qt @ fex2
  c1 = back_substitution(R, y_prime1)
  c2 = back_substitution(R, y_prime2)

  x_poly = np.linspace(a, b, 101)
  y_poly1 = sum(c1[i] * x_poly ** i for i in range(n + 1))
  y_poly2 = sum(c2[i] * x_poly ** i for i in range(n + 1))


  fig, axes = plt.subplots(len(noise_levels), 2, figsize=(12, len(noise_levels) * 4))

  for idx, sigma in enumerate(noise_levels):

    xnoise = normal(N + 1, sigma)
    xeval_noisy = xeval+xnoise
    fex_noise1 = f1(xeval_noisy)
    fex_noise2 = f2(xeval_noisy)

    # noisy input design matrix
    Mnoisy = create_M(xeval_noisy, n)
    Qnoisy, Rnoisy = householder_qr(Mnoisy)
    Qtnoisy = np.transpose(Qnoisy)

    # Solve for coefficients (with input noise)
    y_prime_noise1 = Qtnoisy @ fex_noise1
    y_prime_noise2 = Qtnoisy @ fex_noise2
    c_noise1 = back_substitution(R, y_prime_noise1)
    c_noise2 = back_substitution(R, y_prime_noise2)

    # Generate polynomial values using the coefficients
    y_poly_noise1 = sum(c_noise1[i] * x_poly ** i for i in range(n + 1))
    y_poly_noise2 = sum(c_noise2[i] * x_poly ** i for i in range(n + 1))

    # Plot for Function 1
    axes[idx, 0].scatter(xeval, fex1, color='black', label=r'Function $f(x) = x^5 - 3x^3 + 5x$')
    axes[idx, 0].scatter(xeval, fex_noise1, color='red', label=r'Function with input noise')
    axes[idx, 0].plot(x_poly, y_poly1, color='green', label=f'Polynomial approximation (sigma={sigma})')
    axes[idx, 0].plot(x_poly, y_poly_noise1, color='orange', label=f'Polynomial approximation (sigma={sigma}) with input noise')
    #axes[idx, 0].set_yscale('log')
    axes[idx, 0].legend()
    axes[idx, 0].set_xlabel("x")
    axes[idx, 0].set_ylabel("f(x) / Polynomial Approximation")
    axes[idx, 0].set_title(r"$f(x) = x^5 - 3x^3 + 5x$ with Noise Level " + f"{sigma}")

    # Plot for Function 2
    axes[idx, 1].scatter(xeval, fex2, color='black', label=r'Function $f(x) = x e^{-x^2}$')
    axes[idx, 1].scatter(xeval, fex_noise2, color='red', label=r'Function with input noise')
    axes[idx, 1].plot(x_poly, y_poly2, color='green', label=f'Polynomial approximation (sigma={sigma})')
    axes[idx, 1].plot(x_poly, y_poly_noise2, color='orange', label=f'Polynomial approximation (sigma={sigma}) with input noise')
    axes[idx, 1].legend()
    axes[idx, 1].set_xlabel("x")
    axes[idx, 1].set_ylabel("f(x) / Polynomial Approximation")
    axes[idx, 1].set_title(r"$f(x) = x e^{-x^2}$ with Noise Level " + f"{sigma}")

  # Save the plot
  plt.tight_layout()
  plt.savefig("xnoise_compare_levels.png", dpi=300)
  plt.show()

driver()
