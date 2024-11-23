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
    # fixed polynomial degree to balance fit accuracy and noise robustness 
    degree = 5
    # noise levels to test
    noise_levels = [0.05, 0.1, 0.2, 0.3]

    # function to approximate
    f = lambda x: x * np.exp(-x**2)

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
    Q, R = householder_qr(M)
    Qt = np.transpose(Q)

    # solve for noiseless approximation
    y_obs_noiseless = Qt @ f_obs
    c_noiseless = back_substitution(R, y_obs_noiseless)
    y_noiseless = sum(c_noiseless[i] * x_full**i for i in range(degree + 1)) 

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  

    for idx, sigma in enumerate(noise_levels):
        
        # add noise in x 
        noise = normal(N + 1, sigma)
        f_obs_noisy_y = f_obs + noise
        x_obs_noisy = x_obs + noise 
        f_obs_noisy_x = f(x_obs_noisy)

        # noisy input approximation
        Mnoisy = create_M(x_obs_noisy, degree)
        Qnoisy, Rnoisy = householder_qr(Mnoisy)
        Qtnoisy = np.transpose(Qnoisy)
        y_obs_noisy_x = Qtnoisy @ f_obs_noisy_x
        c_noisy_x = back_substitution(R, y_obs_noisy_x)
        y_poly_noise1_x = sum(c_noisy_x[i] * x_full**i for i in range(degree + 1))

        # noisy output approximation
        y_obs_noisy_y = Qt @ f_obs_noisy_y
        c_noisy_y = back_substitution(R, y_obs_noisy_y)
        y_poly_noise1_y = sum(c_noisy_y[i] * x_full**i for i in range(degree + 1))

        # plot 
        axes[idx].plot(x_full, f_full, color='black', label='True function $f(x) = x e^{-x^2}$')
        #axes[idx].scatter(x_obs, f_obs, color='blue', label='Function samples')
        #axes[idx].scatter(x_obs, f_obs_noisy_x, color='green', label='Function samples with noise in x')
        #axes[idx].scatter(x_obs, f_obs_noisy_y, color='purple', label='Function samples with noise in y')

        # plot observation points
        axes[idx].plot(x_full, y_noiseless, color='red', linestyle='--', label='Noiseless Approximation')
        axes[idx].plot(x_full, y_poly_noise1_x, color='green', linestyle='--', alpha=0.5, label='Noisy x approximation')
        axes[idx].plot(x_full, y_poly_noise1_y, color='orange', linestyle='--', alpha=0.5, label='Noisy y approximation')
        axes[idx].set_xlabel("x")
        axes[idx].set_ylabel("f(x) / Polynomial Approximation")
        axes[idx].set_title(f"Test Interval [1, 2] - Noise Level σ={sigma}")
        axes[idx].legend()
        axes[idx].set_yscale('log')


    plt.tight_layout()
    plt.savefig("interval_test.png", dpi=300)
    plt.show()

driver()
