"""
Write an algorithm that, given a function f(x), number of points n, interval [a, b] and
noise model of your choice, generates noise εi and then data yi = f(xi) + εi
. Plot both
the true function (with a continuous line) and the noisy data (with dots) for each noise
model. What do you observe?
"""

import numpy as np
import matplotlib.pyplot as plt

def generate_noisy_data(f, n, a, b, noise_model):
    """
    Generate noisy data based on a function f(x).
    
    Parameters:
        f (function): mathematical function to evaluate
        n (int): number of points
        a (float): start of the interval
        b (float): end of the interval
        noise_model (str): type of noise ('uniform' or 'gaussian')
        noise_scale (float): Scaling factor for noise level
        
    Returns:
        x (ndarray): x-values
        y_true (ndarray): true y-values
        y_noisy (ndarray): noisy y-values
    """
  
    x = np.linspace(a, b, n)
    y_true = f(x)
    
    # generate noise with reproducibility 
    np.random.seed(42)  
    noise = noise_model(paramters)
    y_noisy = y_true + noise
    
    return x, y_true, y_noisy

# Generate data for two noise models

def f(x):
    return x
  
n, a, b = 50, 0, 2 * np.pi
x_uniform, y_true_uniform, y_noisy_uniform = generate_noisy_data(f, n, a, b, uniform)
x_gaussian, y_true_gaussian, y_noisy_gaussian = generate_noisy_data(f, n, a, b, gaussian)

# Plot results

# Uniform noise
plt.subplot(1, 2, 1)
plt.plot(x_uniform, y_true_uniform, label="True Function", color="blue", linewidth=2)
plt.scatter(x_uniform, y_noisy_uniform, label="Noisy Data (Uniform)", color="red", alpha=0.7)
plt.title("Uniform Noise")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

# Gaussian noise
plt.subplot(1, 2, 2)
plt.plot(x_gaussian, y_true_gaussian, label="True Function", color="blue", linewidth=2)
plt.scatter(x_gaussian, y_noisy_gaussian, label="Noisy Data (Gaussian)", color="green", alpha=0.7)
plt.title("Gaussian Noise")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()

plt.show()
