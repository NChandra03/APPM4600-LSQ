import numpy as np
import matplotlib.pyplot as plt

def create_M(x_values, n):
    """
    Construct the design matrix M for polynomial fitting.
    
    Parameters:
    x_values (ndarray): Array of x-values (of length N) for the data points.
    n (int): The order of the polynomial (degree).
    
    Returns:
    ndarray: The N x (n + 1) design matrix M.
    """
    N = len(x_values)
    M = np.zeros((N, n + 1))
    for i in range(n + 1):
        M[:, i] = x_values ** i  # x^i for each column
    return M

def householder_qr(M, tol=1e-10):
    n, m = M.shape
    Q = np.eye(n)  # Initialize Q as an identity matrix of size n
    R = M.copy()   # Copy of M to transform into R

    for k in range(m):
        # Extract the k-th column vector below the diagonal
        x = R[k:, k]
        norm_x = np.linalg.norm(x)
        
        # Construct the Householder vector u
        u = x.copy()
        u[0] += np.sign(x[0]) * norm_x
        u = u / np.linalg.norm(u)
        
        # Construct the Householder transformation matrix H_k
        H_k = np.eye(n - k) - 2 * np.outer(u, u)
        
        # Embed H_k into the larger identity matrix H
        H = np.eye(n)
        H[k:, k:] = H_k
        
        # Apply the transformation to R and accumulate it in Q
        R = H @ R
        Q = Q @ H.T

    # Truncate R to the shape of the economic version
    R = R[:m, :]

    # Set very small values in R to zero for numerical stability
    #R[np.abs(R) < tol] = 0.0
    
    # Zero out all entries below the diagonal in R
    R = np.triu(R[:m, :])

    return Q[:, :m], R

def back_substitution(R, y_prime, tol=1e-10):
    n = R.shape[0]
    c = np.zeros_like(y_prime, dtype=np.float64)
    # print(R)
    # print(y_prime)
    for i in range(n - 1, -1, -1):
        if abs(R[i, i]) > tol:  # Only proceed if R[i, i] is not effectively zero
            c[i] = (y_prime[i] - np.dot(R[i, i+1:], c[i+1:])) / R[i, i]
        else:
            # If R[i, i] is zero, set c[i] to 0 or handle as desired
            c[i] = 0  # This might not be unique; adjust as needed for your problem
    return(c)

def var_weights(y, min_var=1e-8, max_weight=1e8):
    '''
    Generate error with a proportional dependence on y 
    variance = |yi|

    Inputs: 
        y (np.array): Array of points for variance to be dependent on y
        min_var (int): Threshold for determining an important point
        max_weight (int): Weight to give to important points

    Outputs:
        error (np.array): Error to be added at each point
        weights (np.array): Weights for each point
    '''

    error = []
    variance = np.abs(y)
    for i in range(len(y)):
        error.append(np.random.normal(0,np.sqrt(variance[i])))  

    # Assign minimum and maximum values for when weight function approaces infinity
    # When variance hits the minimum threshold or is 0, assign it a large weight. If not, assign it according to the formula wi = 1/sigma_i
    weights = np.array([
        1 / v**2 if v > min_var else max_weight for v in variance
    ])
    print("list of variance: ", variance)
    print()

    print("list of weights:", weights)

    return np.array(error), weights

def driver():
    np.random.seed(28)

    # Function to approximate
    f = lambda x: x**5 - 3*x**3 + 5*x
    
    # Interval of interest
    a, b = 0, 2
    # Order of approximation
    n = 5
    # Number of points to sample in [a, b]
    N = 100
    xeval = np.linspace(a, b, N + 1)
    fex = f(xeval)
    
    # Generate noise based on y
    error, weights = var_weights(fex)
    # print("list of weights: ", weights)
    noisy_fex = fex + error

    # Create design matrix M
    M = create_M(xeval, n)
    W = np.diag(weights)  # Create diagonal weight matrix
    D = np.diag(np.sqrt(weights)) # Create D
    Mp = D @ M   # Weight the design matrix
    fp = D @ noisy_fex  # Weight the function values

    # Perform weighted QR decomposition
    Qw, Rw = householder_qr(Mp)
    Qtw = np.transpose(Qw)

    # Perform QR decomposition
    Q, R = householder_qr(M)
    Qt = np.transpose(Q)

    # Project f(x) onto the weighted column space of Q and solve for coefficients
    y_primew = Qtw @ fp
    cw = back_substitution(Rw, y_primew)

    # Project f(x) onto the column space of Q and solve for coefficients
    y_prime = Qt @ noisy_fex
    c = back_substitution(R, y_prime)

    # Generate weighted polynomial values using the coefficients
    x_poly = np.linspace(a, b, 100)
    y_polyw = sum(cw[i] * x_poly ** i for i in range(n + 1))

    # Generate polynomial values using the coefficients
    y_poly = sum(c[i] * x_poly ** i for i in range(n + 1))

    # Plot original function, noisy samples, and polynomial approximation
    fig, axes = plt.subplots(1,2)

    axes[0].scatter(xeval, noisy_fex, color='blue', label='Noisy function samples')
    axes[0].plot(x_poly, y_polyw, color='red', label='Weighted Polynomial Approximation')
    axes[0].plot(x_poly, f(x_poly), color='green', linestyle='--', label='True Function')
    axes[0].legend()
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x) / Polynomial Approximation")
    axes[0].set_title("Weighted Polynomial Approximation of x^5 - 3x^3 + 5x")
    
    # Plot original function values and the polynomial approximation
    axes[1].scatter(xeval, noisy_fex, color='blue', label='Noisy function samples')
    axes[1].plot(x_poly, y_poly, color='red', label='Polynomial approximation')
    axes[1].plot(x_poly, f(x_poly), color='green', linestyle='--', label='True Function')
    axes[1].legend()
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f(x) / Polynomial Approximation")
    axes[1].set_title("Polynomial Approximation of x^5 - 3x^3 + 5x")

    plt.show()

driver()
