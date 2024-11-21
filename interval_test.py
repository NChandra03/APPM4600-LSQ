import numpy as np
import matplotlib.pyplot as plt

from noise_func import uniform, normal, generate_noisy_data
from solver_noise import create_M, householder_qr, back_substitution

def driver():
    # num of points to sample
    N = 100
    # observation interval
    a_obs = 0
    b_obs = 1
    # test interval
    a_test = 1
    b_test = 2
    # fixed polynomial degree
    degree = 5
    # noise levels to test
    noise_levels = [0.05, 0.1, 0.2, 0.3]

    # function to approximate
    f = lambda x: x*np.exp(-x**2)

    # observation points
    x_obs = np.linspace(a_obs, b_obs, N + 1)
    f_obs = f(x_obs)

    # matrix for observations
    M_obs = create_M(x_obs, degree)
  
    # perform QR decomposition on the observation matrix
    Q_obs, R_obs = householder_qr(M_obs)
    Qt_obs = np.transpose(Q_obs)

    # solve for noiseless approximation
    y_obs_noiseless = Qt_obs @ f_obs
    c_noiseless = back_substitution(R_obs, y_obs_noiseless)
    y_test_noiseless = sum(c_noiseless[i] * x_test ** i for i in range(degree + 1))

    # test points
    x_test = np.linspace(a_test, b_test, 101)

    fig, ax = plt.subplots(len(noise_levels), 1, figsize=(10, len(noise_levels) * 4))

    for idx, sigma in enumerate(noise_levels):
        # add noise to observations
        noise = normal(N + 1, sigma)
        f_obs_noisy = f_obs + noise
        x_test_noisy = x_test + noise

        # solve for noisy approximation 
        y_obs_noisy = Qt_obs @ f_obs_noisy
        c_noisy = back_substitution(R_obs, y_obs_noisy)
        y_test_noisy = sum(c_noisy[i] * x_test ** i for i in range(degree + 1))

        # plot 
        ax[idx].plot(x_test, f(x_test), color='black', label='Original Function')
        ax[idx].plot(x_test, y_test_noiseless, color='red', label='Noiseless Approximation')
        ax[idx].plot(x_test_noisy, y_test_noiseless, color='green', linestyle='--', label=f'Noisy x Approximation (σ={sigma})')
        ax[idx].plot(x_test, y_test_noisy, color='blue', linestyle='--', label=f'Noisy y Approximation (σ={sigma})')
        ax[idx].plot(y_test_noisy, y_test_noisy, color='orange', linestyle='--', label=f'Noisy x & y Approximation (σ={sigma})')
        ax[idx].set_xlabel("x")
        ax[idx].set_ylabel("f(x) / Polynomial Approximation")
        ax[idx].set_title(f"Test Interval [1, 2] - Noise Level σ={sigma}")
        ax[idx].legend()
      
    plt.tight_layout()
    plt.savefig("noise_comparison_test_interval.png", dpi=300)
    plt.show()

driver()
