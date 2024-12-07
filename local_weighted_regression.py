import numpy as np
import matplotlib.pyplot as plt
from noise_func import uniform, normal, generate_noisy_data
from solver_noise import create_M, householder_qr, back_substitution
from WLSQ_working import weighted_least_squares


def driver():
  # number of points to sample
  N = 100
  # interval of interest
  a = 0
  b = 2

  # functions to approximate
  #f1 = lambda x: x**5 - 3*x**3 + 5*x
  f1 = lambda x: x * np.exp(-x**2)
  f2 = lambda x: x * np.exp(-x**2)

  # sample points
  xeval = np.linspace(a, b, N + 1)
  fex1 = f1(xeval)
  fex2 = f2(xeval)

  # choose order for noise level analysis
  n = 5
  noise_levels = [0.05, 0.1, 0.2, 0.3]

  # noiseless input design matrix
  M = create_M(xeval, n)
  Q, R = householder_qr(M)
  Qt = np.transpose(Q)

  # solve for coefficients (noise-free)
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
    fex_smooth1 = local_smooth(xeval, fex_noise1, (b-a)/10, 0.5, 1)
    fex_smooth2 = local_smooth(xeval, fex_noise2, (b-a)/10, 0.5, 2)

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

    # linear local weighted regression
    axes[idx, 0].scatter(xeval, fex1, color='black', label=r"Original data: $f(x) = x e^{-x^2}$")
    axes[idx, 0].scatter(xeval, fex_noise1, color='grey', label=r"Noisy data (\sigma = " + f"{sigma}")
    axes[idx, 0].scatter(xeval, fex_smooth1, color='red', label='Linearly smoothed data')
    axes[idx, 0].plot(x_poly, y_poly1, color='green', label='LSR approximation of original data')
    axes[idx, 0].plot(x_poly, y_poly_noise1, color='blue', label='LSR approximation of noisy data')
    axes[idx, 0].plot(x_poly, y_poly_smooth1, color='orange', label='LSR approximation of smoothed data')
    #axes[idx, 0].set_yscale('log')
    axes[idx, 0].legend()
    axes[idx, 0].set_xlabel("x")
    axes[idx, 0].set_ylabel("f(x) / Polynomial Approximation")
    #axes[idx, 0].set_title(r"$f(x) = x e^{-x^2}$ with Noise Level " + f"{sigma}")

    # quadratic local weighted regression
    axes[idx, 1].scatter(xeval, fex2, color='black', label=r"Original data: $f(x) = x e^{-x^2}$")
    axes[idx, 1].scatter(xeval, fex_noise2, color='grey', label=r"Noisy data (\sigma = " + f"{sigma}")
    axes[idx, 1].scatter(xeval, fex_smooth2, color='red', label='Quadratically smoothed data')
    axes[idx, 1].plot(x_poly, y_poly2, color='green', label=f'LSR approximation of original data')
    axes[idx, 1].plot(x_poly, y_poly_noise2, color='blue', label='LSR approximation of noisy data')
    axes[idx, 1].plot(x_poly, y_poly_smooth2, color='orange', label='LSR approximation of smoothed data')
    axes[idx, 1].legend()
    axes[idx, 1].set_xlabel("x")
    axes[idx, 1].set_ylabel("f(x) / Polynomial Approximation")
    #axes[idx, 1].set_title(r"$f(x) = x e^{-x^2}$ with Noise Level " + f"{sigma}")

  plt.tight_layout()
  plt.savefig("compare_lsr_smoothing.png", dpi=300)
  plt.show()
'''

def driver():
    # Set parameters for testing
    n_points = 250
    noise_level = 20  # High noise to test the smoothing
    bandwidth = 0.3  # Bandwidth for local smoothing
    degree = 2  # Quadratic polynomial for local regression

    # Generate noisy data
    xi, yi_true, yi_noisy = generate_noisy_data(n_points, noise_level)

    # Apply the local smoothing function
    smoothed_yi = local_smooth(xi, yi_noisy, bandwidth, degree)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(xi, yi_true, label='True Function', color='green', linewidth=2)
    plt.scatter(xi, yi_noisy, label='Noisy Data', color='red', alpha=0.6)
    plt.plot(xi, smoothed_yi, label='Smoothed Data (Local Regression)', color='blue', linewidth=2)
    plt.title("Local Weighted Regression on Noisy Data")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.show()
    plt.savefig("test.png")

'''

# Generate a noisy dataset
def generate_noisy_data(n_points=1000, noise_level=5.0):
    # Generate x-values from a range
    xi = np.linspace(-10, 10, n_points)
    
    # True function: quadratic function (for example)
    yi_true = 2 * xi**2 + 3 * xi + 5  # y = 2x^2 + 3x + 5
    
    # Add random Gaussian noise to the true y-values
    yi_noisy = yi_true + np.random.normal(scale=noise_level, size=n_points)
    
    return xi, yi_true, yi_noisy


def tri_weight(u):
    """Tukey tri-weight kernel function."""
    if abs(u) <= 1:
        return (1 - abs(u)**3)**3
    return 0

def local_smooth(xi, yi, bandwidth, span, degree):
    """
    Smooth data using local linear weighted regression with a specified bandwidth.
    Parameters:
        xi (array): the x-coordinates of the data points.
        yi (array): the y-coordinates of the data points.
        bandwidth (float): the bandwidth (window size in x) for smoothing.
        span (float): The proportion of points to use in the local fit.
        degree (int): The degree of the local polynomial (1 for linear, 2 for quadratic).
    Returns:
        smoothed_xi (array): the smoothed x-coordinates.
        smoothed_yi (array): the smoothed y-coordinates.
    """
    smoothed_yi = []
    
    N = len(xi)
    num_neighbors = int(span * N)  # Number of neighbors to use in the local fit
    
    for i in range(N):

        weights = []
        x_local_diff = []
        x_local = []
        y_local = []

        # get neighbors based on subinterval spa]
        distances = np.abs(xi - xi[i])
        sorted_indices = np.argsort(distances)
        nearest_indices = sorted_indices[:num_neighbors]

        # get local (subinterval) data points
        x_local = xi[nearest_indices]
        y_local = yi[nearest_indices]

        # center x_local around xi[i]
        x_local_diff = x_local - xi[i]

        # compute weights using triangular kernel
        u = x_local_diff / bandwidth
        for j in range(len(u)):
            weights.append(tri_weight(u[j]))

        # construct design matrix for the local regression (based on degree)
        M_local = create_M(np.array(x_local_diff), degree)
        
        # solve wls to find the coefficients of linear or quadratic polynomial (betas)
        c_local = weighted_least_squares(M_local, np.array(y_local), np.array(weights))

        # polynomial evaluated at xj = xi gives smoothed value (just constant term)
        y_smoothed = 0
        for k in range(len(c_local)):
            y_smoothed += c_local[k] * (xi[i] - xi[i])**k  

        smoothed_yi.append(y_smoothed)
    
    return np.array(smoothed_yi)

driver()