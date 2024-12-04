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
    variance = 100 * np.abs(y)
    for i in range(len(y)):
        error.append(np.random.normal(0,np.sqrt(variance[i])))  

    # Assign minimum and maximum values for when weight function approaces infinity
    # When variance hits the minimum threshold or is 0, assign it a large weight. If not, assign it according to the formula wi = 1/sigma_i
    weights = np.array([
        1 / v**2 if v > min_var else max_weight for v in variance
    ])
    #print("list of variance: ", variance)
    #print("list of weights:", weights)

    return np.array(error), weights, variance


def driver():
    np.random.seed(28)

    # Function to approximate
    # Functions to approximate
    f1 = lambda x: x**5 - 3*x**3 + 5*x 
    f2 = lambda x: x*np.exp(-x**2)

    # Interval of interest
    a, b = 0, 2
    # Order of approximation
    degrees = [1,3,5]

    # Number of points to sample in [a, b]
    N = 100
    xeval = np.linspace(a, b, N + 1)
    fex1 = f1(xeval)
    fex2 = f2(xeval)
    
    # Generate noise based on y
    error1, weights1, variance1 = var_weights(fex1)
    error2, weights2, variance2 = var_weights(fex2)

    # adjust endpoint weights
    #weights1[-3:] = [1e6, 1e6, 1e6]
    #weights2[-3:] = [1e6, 1e6, 1e6]

    noisy_fex1 = fex1 + error1
    noisy_fex2 = fex2 + error2

    fig, axes = plt.subplots(len(degrees), 2, figsize=(12, len(degrees) * 4))


    for i, n in enumerate(degrees):
   
        # Create design matrix M
        M = create_M(xeval, n)
        W1 = np.diag(weights1)  # Create diagonal weight matrix
        W2 = np.diag(weights2)  # Create diagonal weight matrix
        D1 = np.diag(np.sqrt(weights1)) # Create D
        D2 = np.diag(np.sqrt(weights2)) # Create D
        Mp1 = D1 @ M   # Weight the design matrix
        Mp2 = D2 @ M   # Weight the design matrix
        fp1 = D1 @ noisy_fex1  # Weight the function values
        fp2 = D2 @ noisy_fex2  # Weight the function values

        # Perform weighted QR decomposition
        Qw1, Rw1 = householder_qr(Mp1)
        Qw2, Rw2 = householder_qr(Mp2)
        Qtw1 = np.transpose(Qw1)
        Qtw2 = np.transpose(Qw2)

        # Perform QR decomposition
        Q, R = householder_qr(M)
        Qt = np.transpose(Q)

        # Project f(x) onto the weighted column space of Q and solve for coefficients
        y_primew1 = Qtw1 @ fp1
        y_primew2 = Qtw2 @ fp2
        cw1 = back_substitution(Rw1, y_primew1)
        cw2 = back_substitution(Rw2, y_primew2)

        # Project f(x) onto the column space of Q and solve for coefficients
        y_prime1 = Qt @ noisy_fex1
        y_prime2 = Qt @ noisy_fex2
        c1 = back_substitution(R, y_prime1)
        c2 = back_substitution(R, y_prime2)

        # Generate weighted polynomial values using the coefficients
        x_poly = np.linspace(a, b, 100)
        y_polyw1 = sum(cw1[i] * x_poly ** i for i in range(n + 1))
        y_polyw2 = sum(cw2[i] * x_poly ** i for i in range(n + 1))

        # Generate polynomial values using the coefficients
        y_poly1 = sum(c1[i] * x_poly ** i for i in range(n + 1))
        y_poly2 = sum(c2[i] * x_poly ** i for i in range(n + 1))

        #for i in range(100):
            #print('variance: ', variance[i], ', weight: ' , weights[i], ', approximation unweighted: ', y_poly[i], ', approximation weighted: ' , y_polyw[i], 'true value: ', f(i))


        # Plot for Function 1
        axes[i,0].scatter(xeval, noisy_fex1, color='blue', alpha=0.3, label='Noisy function samples')
        axes[i,0].plot(x_poly, y_poly1, color='orange', label='Unweighted polynomial approximation')
        axes[i,0].plot(x_poly, y_polyw1, color='red', label='Weighted Polynomial Approximation')
        axes[i,0].plot(x_poly, f1(x_poly), color='green', linestyle='--', label='True Function')
        #axes[i, 0].set_yscale('log')
        axes[i, 0].legend()
        axes[i, 0].set_xlabel("x")
        axes[i, 0].set_ylabel("f(x) / Polynomial Approximation")
        axes[i, 0].set_title(r"$f(x) = x^5 - 3x^3 + 5x$ with Degree " + f"{n}" + " Polynomial Approximation")

        # Plot for Function 2
        axes[i, 1].scatter(xeval, noisy_fex2, color='blue', alpha=0.3, label='Noisy function samples')
        axes[i, 1].plot(x_poly, y_poly2, color='orange', label='Unweighted polynomial approximation')
        axes[i, 1].plot(x_poly, y_polyw2, color='red', label='Weighted Polynomial Approximation')
        axes[i, 1].plot(x_poly, f2(x_poly), color='green', linestyle='--', label='True Function')
        axes[i, 1].legend()
        axes[i, 1].set_xlabel("x")
        axes[i, 1].set_ylabel("f(x) / Polynomial Approximation")
        axes[i, 1].set_title(f"Function 2: Degree {n}")
        axes[i, 1].set_title(r"$f(x) = x e^{-x^2}$ with Degree " + f"{n}" + " Polynomial Approximation")

    plt.tight_layout()
    plt.show()
    plt.savefig("weighted.png", dpi=300)
    '''

    plt.scatter(xeval, noisy_fex, color='blue', alpha=0.3, label='Noisy function samples')
    plt.plot(x_poly, y_poly, color='orange', label='Unweighted polynomial approximation')
    plt.plot(x_poly, y_polyw, color='red', label='Weighted Polynomial Approximation')
    plt.plot(x_poly, f(x_poly), color='green', linestyle='--', label='True Function')
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x) / Polynomial Approximation")
    plt.title("Weighted Polynomial Approximation of x^5 - 3x^3 + 5x")

    
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
    plt.savefig('weighted.png')
    '''


driver()
