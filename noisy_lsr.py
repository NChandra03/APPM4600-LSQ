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

def driver():
  
  n = 50
  a = 0
  b = 2*np.pi

  # first function
  f1 = lambda x: x**5 - 3x**3 + 5x

  # second function
  f2 = lambda x: x*np.exp(-x**2)
  
  x_uniform, y_true_uniform, y_noisy_uniform = generate_noisy_data(f, n, a, b, uniform)
  x_gaussian, y_true_gaussian, y_noisy_gaussian = generate_noisy_data(f, n, a, b, gaussian)
  
  # Plot results
  
  plt.plot(x_uniform, y_true_uniform, label="True Function: f(x)=sin(x)", color="blue", linewidth=2)
  plt.scatter(x_uniform, y_noisy_uniform, label="Noisy Data (Uniform)", color="red")
  plt.scatter(x_gaussian, y_noisy_gaussian, label="Noisy Data (Gaussian)", color="green")
  plt.title("Noise Model")
  plt.xlabel("x")
  plt.ylabel("y")
  plt.legend()
  
  plt.show()
  plt.savefig("generated_noise.png")
  


def uniform(n,minimum,maximum):
    error = []
    for i in range (n):
        # centering noise at 0, range is  -(b - a)/2 to (b - a)/2 
        error.append(random.random() * (maximum - minimum) - (maximum - minimum) / 2)

    return(error)

def gaussian(n,sigma):
    mu = 0
    error = []
    for i in range (n):
        error.append(np.random.normal(mu, sigma))

    return(error)

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

    # variance in gaussian model is sigma^2
    # variance in uniform model is (maximum-minimum)^2/12
    # setting these equal, we see that the maximum-minimum is equal to sqrt(12)*sigma
    # to center the uniform noise around 0, the minimum can be written as -sqrt(12)*sigma
    # and the maximum can be written as +sqrt(12)*sigma

    sigma = 0.5
    minimum = -np.sqrt(12) * sigma /2
    maximum = np.sqrt(12) * sigma /2

    # generate noise with reproducibility 
    np.random.seed(42)
    if noise_model == uniform:
        noise = uniform(n,minimum,maximum)
        y_noisy = y_true + noise
        return x, y_true, y_noisy

    elif noise_model == gaussian:
        noise = gaussian(n,sigma)
        y_noisy = y_true + noise
        return x, y_true, y_noisy

    else:
        print("noise model not recognized")
        return 

driver()
