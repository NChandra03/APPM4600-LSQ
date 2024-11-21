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
    # fixed polynomial degree to balance fit accuracy and and noise robustness 
    degree = 10
    # noise levels to test
    noise_levels = [0.05, 0.1, 0.2, 0.3]

    # function to approximate
    f = lambda x: x*np.exp(-x**2)

    # observation points
    x_obs = np.linspace(a_obs, b_obs, N + 1)
    f_obs = f(x_obs)

    # test points
    x_test = np.linspace(a_test, b_test, 101)
    f_test = f(x_test)

    # full interval for plot
    x_full = np.linspace(a_obs, b_test, 101)
    f_full = f(x_full)


    # build matrix with observations
    M = create_M(x_obs, degree)
  
    # perform QR decomposition on the observation matrix
    Q,R = householder_qr(M)
    Q = np.transpose(Q)

    # solve for noiseless approximation
    y_obs_noiseless = Q @ f_obs
    c_noiseless = back_substitution(R, y_obs_noiseless)
    # evaluate polynomial approx on test interval 
    y_test_noiseless = sum(c_noiseless[i] * x_test ** i for i in range(degree + 1)) 

    fig, axes = plt.subplots(len(noise_levels), 1, figsize=(10, len(noise_levels) * 4))

    for idx, sigma in enumerate(noise_levels):
        # add noise to observations
        noise = normal(N + 1, sigma)
        f_obs_noisy = f_obs + noise
        x_test_noisy = x_test + noise

        x_obs_noisy = x_obs + noise 

        # now solve for noisy approximation 
        y_obs_noisy = Q @ f_obs_noisy
        c_noisy = back_substitution(R, y_obs_noisy)
        y_test_noisy = sum(c_noisy[i] * x_test ** i for i in range(degree + 1))

        # plot true function on full interval
        axes[idx].plot(x_full, f_full, color='black', label='Original Function')

        # plot observation points
        
        axes[idx].plot(x_obs, y_test_noiseless, color='red', alpha=0.5, label='Noiseless Approximation')
        axes[idx].plot(x_obs_noisy, y_test_noiseless, color='green', linestyle='--', alpha=0.5, label=f'Noisy x Approximation (σ={sigma})')
        axes[idx].plot(x_obs, y_test_noisy, color='blue', linestyle='--', alpha=0.5, label=f'Noisy y Approximation (σ={sigma})')
        axes[idx].plot(x_obs_noisy, y_test_noisy, color='orange', linestyle='--', alpha=0.5, label=f'Noisy x & y Approximation (σ={sigma})')
        
        # plot test points 
        '''
        axes[idx].plot(x_test, y_test_noiseless, color='red', label='Noiseless Approximation')
        axes[idx].plot(x_test_noisy, y_test_noiseless, color='green', linestyle='--', label=f'Noisy x Approximation (σ={sigma})')
        axes[idx].plot(x_test, y_test_noisy, color='blue', linestyle='--', label=f'Noisy y Approximation (σ={sigma})')
        axes[idx].plot(x_test_noisy, y_test_noisy, color='orange', linestyle='--', label=f'Noisy x & y Approximation (σ={sigma})')
        '''

        axes[idx].set_xlabel("x")
        axes[idx].set_ylabel("f(x) / Polynomial Approximation")
        axes[idx].set_title(f"Test Interval [1, 2] - Noise Level σ={sigma}")
        axes[idx].legend()
        axes[idx].set_yscale('log')

      
    plt.tight_layout()
    plt.savefig("obs_interval_noise_comparison.png", dpi=300)
    plt.show()

driver()
