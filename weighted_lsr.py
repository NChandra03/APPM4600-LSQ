import numpy as np
import matplotlib.pyplot as plt

def create_M(x_values, n):
    """
    Construct the design matrix M for polynomial fitting.
    """
    N = len(x_values)
    M = np.zeros((N, n + 1))
    for i in range(n + 1):
        M[:, i] = x_values ** i  # x^i for each column
    return M

def householder_qr(M, tol=1e-10):
    """
    Perform QR decomposition using Householder reflections.
    """
    n, m = M.shape
    Q = np.eye(n)  # Initialize Q as an identity matrix of size n
    R = M.copy()   # Copy of M to transform into R

    for k in range(m):
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        u = x.copy()
        u[0] += np.sign(x[0]) * norm_x
        u = u / np.linalg.norm(u)
        H_k = np.eye(n - k) - 2 * np.outer(u, u)
        H = np.eye(n)
        H[k:, k:] = H_k
        R = H @ R
        Q = Q @ H.T

    R = np.triu(R[:m, :])  # Ensure R is upper triangular
    return Q[:, :m], R

def back_substitution(R, y_prime, tol=1e-10):
    """
    Solve Rx = y' for x using back substitution.
    """
    n = R.shape[0]
    c = np.zeros_like(y_prime, dtype=np.float64)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) > tol:
            c[i] = (y_prime[i] - np.dot(R[i, i+1:], c[i+1:])) / R[i, i]
        else:
            c[i] = 0
    return c

def lin_var_weights(x, a=0, b=1, min_var=1e-6, max_weight=1e8):
    '''
    Generate error with a linear dependence on variance 
    variance = a + b*xi

    Inputs: 
        x (np.array): Array of points for variance to be dependent on (can be x values or y values)
        a (int): Minimum noise level expected in system (y intercept)
        b (int): Growth rate of variance (slope)

    Outputs:
        error (np.array): Error to be added at each point
    '''

    error = []
    variance = np.abs(a + b*x)
    for i in range(len(x)):
        error.append(np.random.normal(0,np.sqrt(variance[i])))  

    # Assign minimum and maximum values for when weight function approaces infinity
    # When variance hits the minimum threshold or is 0, assign it a large weight. If not, assign it according to the formula wi = 1/sigma_i
    # weights = np.where((variance <= min_var) | (variance == 0), max_weight, 1 / variance**2)
    weights = np.array([
        1 / v**2 if v > min_var else max_weight for v in variance
    ])
    # print("list of variance: ", variance)
    # print()

    # print("list of weights:", weights)

    return np.array(error), weights

def driver():
    # np.random.seed(28)

    # Function to approximate
    f = lambda x: x**5 - 3*x**3 + 5*x
    
    # Interval of interest
    a, b = -3, 3
    # Order of approximation
    n = 5
    # Number of points to sample in [a, b]
    N = 100
    xeval = np.linspace(a, b, N + 1)
    fex = f(xeval)
    
    # Generate noise based on y
    error, weights = lin_var_weights(fex, a=0, b=100)
    # print("list of weights: ", weights)
    noisy_fex = fex + error

    # Weighted design matrix
    M = create_M(xeval, n)
    W = np.diag(weights)  # Create diagonal weight matrix
    MW = np.sqrt(W) @ M   # Weight the design matrix
    fW = np.sqrt(W) @ noisy_fex  # Weight the function values

    # Perform weighted QR decomposition
    Q, R = householder_qr(MW)
    Qt = np.transpose(Q)

    # Solve for coefficients using back substitution
    y_prime = Qt @ fW
    c = back_substitution(R, y_prime)

    # Generate polynomial values using the coefficients
    x_poly = np.linspace(a, b, 100)
    y_poly = sum(c[i] * x_poly ** i for i in range(n + 1))

    # Plot original function, noisy samples, and polynomial approximation
    plt.scatter(xeval, noisy_fex, color='blue', label='Noisy function samples')
    plt.plot(x_poly, y_poly, color='red', label='Weighted Polynomial Approximation')
    plt.plot(x_poly, f(x_poly), color='green', linestyle='--', label='True Function')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x) / Polynomial Approximation")
    plt.title("Weighted Polynomial Approximation of x^5 - 3x^3 + 5x")
    plt.show()

driver()
