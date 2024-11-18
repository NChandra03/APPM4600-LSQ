import numpy as np
import matplotlib.pyplot as plt
import random

def driver():

    f = lambda x: x

    a = 0
    b = 5

    x = np.linspace(a, b, 200)
    y = f(x)

    intercept = 0
    slope = 0.2

    np.random.seed(28)

    figure, axis = plt.subplots(3,2, figsize=(10, 30))

    axis[0,0].plot(x, y, label="True function f(x) = x")
    axis[0,0].scatter(x, y + lin_var(x, intercept, slope), label="Linear variance", color="red",s=10)
    axis[0,0].set_title("Linear Variance Dependent on Input")
    axis[0,0].set_xlabel("x")
    axis[0,0].set_ylabel("y")
    axis[0,0].legend()

    axis[0,1].plot(x, y, label="True function f(x) = x")
    axis[0,1].scatter(x, y + lin_var(y, intercept, slope), label="Linear Variance", color="green",s=10)
    axis[0,1].set_title("Linear Variance Dependent on Output")
    axis[0,1].set_xlabel("x")
    axis[0,1].set_ylabel("y")
    axis[0,1].legend()

    axis[1,0].plot(x, y, label="True function f(x) = x")
    axis[1,0].scatter(x, y + quad_var(x, intercept, slope), label="Quadratic variance", color="red",s=10)
    axis[1,0].set_title("Quadratic Variance Dependent on Input")
    axis[1,0].set_xlabel("x")
    axis[1,0].set_ylabel("y")
    axis[1,0].legend()

        

    axis[1,1].plot(x, y, label="True function f(x) = x")
    axis[1,1].scatter(x, y + quad_var(y, intercept, slope), label="Quadratic variance", color="green",s=10)
    axis[1,1].set_title("Quadratic Variance Dependent on Output")
    axis[1,1].set_xlabel("x")
    axis[1,1].set_ylabel("y")
    axis[1,1].legend()

    axis[2,0].plot(x, y, label="True function f(x) = x")
    axis[2,0].scatter(x, y + exp_var(x, 0.1, 1.5), label="Exponential variance", color="red",s=10)
    axis[2,0].set_title("Exponential Variance Dependent on Input")
    axis[2,0].set_xlabel("x")
    axis[2,0].set_ylabel("y")
    axis[2,0].legend()

    axis[2,1].plot(x, y, label="True function f(x) = x")
    axis[2,1].scatter(x, y + exp_var(y, 0.1, 1.5), label="Exponential variance", color="green",s=10)
    axis[2,1].set_title("Exponential Variance Dependent on Output")
    axis[2,1].set_xlabel("x")
    axis[2,1].set_ylabel("y")
    axis[2,1].legend()

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.5, wspace=0.3)
    plt.show()

    return


def lin_var(x, a=1, b=1):
    '''
    Generate error with a linear dependence on variance 
    error = a + bxi*variance

    Inputs: 
        x (np.array): Array of points for variance to be dependent on (can be x values or y values)
        a (int): Minimum noise level expected in system (y intercept)
        b (int): Growth rate of variance (slope)

    Outputs:
        error (np.array): Error to be added at each point
    '''
    # error = []
    # for i in range(len(x)):
    #     variance = np.random.randn()
    #     # normal gaussian distribution centered around 0
    #     error.append(a + b*x[i]*variance)

    error = a + b*x*np.random.randn(len(x))

    return np.array(error)


def quad_var(x, a=1, b=1):
    '''
    Generate error with a quadratic dependence on variance 
    sigma(xi) = a + bxi^2

    Inputs: 
        x (np.array): Array of points for variance to be dependent on (can be x values or y values)
        a (int): Minimum noise level expected in system (y intercept)
        b (int): Growth rate of variance (slope)

    Outputs:
        error (np.array): Error to be added at each point
    '''
    # error = []
    # for i in range(len(x)):
    #     variance = a + b * x[i]**2
    #     # normal gaussian distribution centered around 0
    #     error.append(np.random.normal(0, np.sqrt(variance)))
    error = a + b*x**2*np.random.randn(len(x))

    return np.array(error)

def exp_var(x, a=1, b=1):
    '''
    Generate error with an exponential dependence on variance 
    a * e^(b * x) * variance

    Inputs: 
        x (np.array): Array of points for variance to be dependent on (can be x values or y values)
        a (int): Scale factor 
        b (int): Growth rate of variance 

    Outputs:
        error (np.array): Error to be added at each point
    '''
    error = a * np.exp(b * x) * np.random.randn(len(x))

    return np.array(error)    


driver()