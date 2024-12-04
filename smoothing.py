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

  # Functions to approximate
  f1 = lambda x: x**5 - 3*x**3 + 5*x
  f2 = lambda x: x * np.exp(-x**2)

  # Sample points
  xeval = np.linspace(a, b, N + 1)
  fex1 = f1(xeval)
  fex2 = f2(xeval)

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

    # add smoothing
    x_smooth1, fex_smooth1 = bin_smooth_data(xeval, fex_noise1, (b-a)/10)
    x_smooth2, fex_smooth2 = bin_smooth_data(xeval, fex_noise2, (b-a)/10)

    # noisy input design matrix
    Mnoisy = create_M(xeval_noisy, n)
    Qnoisy, Rnoisy = householder_qr(Mnoisy)
    Qtnoisy = np.transpose(Qnoisy)

    # coefficients (with input noise)
    y_prime_noise1 = Qtnoisy @ fex_noise1
    y_prime_noise2 = Qtnoisy @ fex_noise2
    c_noise1 = back_substitution(R, y_prime_noise1)
    c_noise2 = back_substitution(R, y_prime_noise2)

    # coefficients (with smoothing), use Qt since developed from xeval
    y_prime_smooth1 = Qt @ fex_smooth1
    y_prime_smooth2 = Qt @ fex_smooth2
    c_smooth1 = back_substitution(R, y_prime_smooth1)
    c_smooth2 = back_substitution(R, y_prime_smooth2)

    # polynomial values using the coefficients
    y_poly_noise1 = sum(c_noise1[i] * x_poly ** i for i in range(n + 1))
    y_poly_noise2 = sum(c_noise2[i] * x_poly ** i for i in range(n + 1))
    y_poly_smooth1 = sum(c_smooth1[i] * x_poly ** i for i in range(n + 1))
    y_poly_smooth2 = sum(c_smooth2[i] * x_poly ** i for i in range(n + 1))


    # function 1
    axes[idx, 0].scatter(xeval, fex1, color='black', label='Original data')
    axes[idx, 0].scatter(xeval, fex_noise1, color='grey', label='Noisy data')
    axes[idx, 0].scatter(xeval, fex_smooth1, color='red', label='Smoothed data')
    axes[idx, 0].plot(x_poly, y_poly1, color='green', label='LSR approximation of original data')
    axes[idx, 0].plot(x_poly, y_poly_noise1, color='blue', label='LSR approximation of noisy data')
    axes[idx, 0].plot(x_poly, y_poly_smooth1, color='orange', label='LSR approximation of smoothed data')
    #axes[idx, 0].set_yscale('log')
    axes[idx, 0].legend()
    axes[idx, 0].set_xlabel("x")
    axes[idx, 0].set_ylabel("f(x) / Polynomial Approximation")
    axes[idx, 0].set_title(r"$f(x) = x^5 - 3x^3 + 5x$ with Noise Level " + f"{sigma}")

    # function 2
    axes[idx, 1].scatter(xeval, fex2, color='black', label='Original data')
    axes[idx, 1].scatter(xeval, fex_noise2, color='grey', label='Noisy data')
    axes[idx, 1].scatter(xeval, fex_smooth2, color='red', label='Smoothed data')
    axes[idx, 1].plot(x_poly, y_poly2, color='green', label=f'LSR approximation of original data')
    axes[idx, 1].plot(x_poly, y_poly_noise2, color='blue', label='LSR approximation of noisy data')
    axes[idx, 1].plot(x_poly, y_poly_smooth2, color='orange', label='LSR approximation of smoothed data')
    axes[idx, 1].legend()
    axes[idx, 1].set_xlabel("x")
    axes[idx, 1].set_ylabel("f(x) / Polynomial Approximation")
    axes[idx, 1].set_title(r"$f(x) = x e^{-x^2}$ with Noise Level " + f"{sigma}")

  plt.tight_layout()
  plt.savefig("smooth_compare_noise_levels.png", dpi=300)
  plt.show()


# bin smoothing function 

def bin_smooth_data(xi, yi, bandwidth):
    """
    Smooth data using bin smoothing with a specified bandwidth.
    Parameters:
        xi (array): the x-coordinates of the data points.
        yi (array): the y-coordinates of the data points.
        bandwidth (float): the bandwidth (window size in x) for smoothing.
    Returns:
        smoothed_xi (array): the smoothed x-coordinates.
        smoothed_yi (array): the smoothed y-coordinates.
    """
    smoothed_yi = []
    smoothed_xi = []

    for i in range(len(xi)):
        x_in_window = []
        y_in_window = []

        # find points in bandwidth 
        for j in range(len(xi)):
            if(abs(xi[j]-xi[i])<=bandwidth):
                    x_in_window.append(xi[j])
                    y_in_window.append(yi[j])

        # find the mean x and y on the interval 
        mean_xi = np.mean(x_in_window)
        mean_yi = np.mean(y_in_window)
        smoothed_xi.append(mean_xi)
        smoothed_yi.append(mean_yi)
    
    return np.array(smoothed_xi), np.array(smoothed_yi)

driver()
