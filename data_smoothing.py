import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta
from solver_noise import create_M, householder_qr, back_substitution
from WLSQ_working import weighted_least_squares

def driver():

    # apple stock
    ticker = "AAPL"  
    start_date = "2020-01-01"
    end_date = "2023-12-01"

    # download stock price data
    data = yf.download(ticker, start=start_date, end=end_date)

    # extract x and y values
    x = data.index  # dates / timestamps
    y = data['Close']  # closing prices (stock price recorded at 4 pm ET)

    # bandwidths: 7 and 30 days
    bandwidths = [timedelta(days=30), timedelta(days=90)]

    fig, axs = plt.subplots(4, 2, figsize=(16, 16), sharex=True, sharey=True)

    smoothing_methods = [
        ("Bin Smoothing", bin_smooth_data),
        ("Kernel Smoothing", kernel_smooth_data),
        ("Local Linear Smoothing", lambda xi, yi, bw: local_smooth(xi, yi, bw, 0.5, 1)),
        ("Local Quadratic Smoothing", lambda xi, yi, bw: local_smooth(xi, yi, bw, 0.5, 2)),
    ]

    for row, (title, method) in enumerate(smoothing_methods):
        for col, bandwidth in enumerate(bandwidths):
            # Compute smoothed data
            smoothed_y = method(x, np.array(y), bandwidth)

            # Plot data and smoothed fit
            axs[row, col].plot(x, y, color='grey', label=f'Apple Closing Prices')
            axs[row, col].plot(x, smoothed_y, color='red', label=f'{title} (BW {bandwidth.days} Days)')
            axs[row, col].set_title(f"{title} (Bandwidth {bandwidth.days} Days)")
            axs[row, col].grid()
            if row == 3:  # Add x-axis label to the last row
                axs[row, col].set_xlabel("Date")
            if col == 0:  # Add y-axis label to the first column
                axs[row, col].set_ylabel("Stock Price")
            axs[row, col].legend()

    # Adjust layout and save figure
    fig.tight_layout()
    fig.savefig("data_smoothed_compare_bandwidth.png")
    plt.close(fig)

    # separate figure overlaying all smoothing methods
    fig_overlay, ax = plt.subplots(figsize=(12, 8))
    plt.plot(x, y, color='blue', label=f'{ticker} Closing Prices')
    plt.plot(x, bin_smooth_data(x, np.array(y), bandwidths[0]), color='red', linestyle='-', label='Bin Smoothing')
    plt.plot(x, kernel_smooth_data(x, np.array(y), bandwidths[0]), color='orange', linestyle='--', label='Kernel Smoothing')
    plt.plot(x, local_smooth(x, np.array(y), bandwidths[0], 0.5, 1), color='green', linestyle='-.', label='Local Linear Smoothing')
    plt.plot(x, local_smooth(x, np.array(y), bandwidths[0], 0.5, 2), color='blue', linestyle=':', label='Local Quadratic Smoothing')

    plt.title(f'{ticker} Stock Price with Smoothing Fits')
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()
    plt.savefig("data_smoothed_compare.png")
    plt.close(fig_overlay)

    '''
    
    bandwidth = timedelta(days=7)  


    bin_smooth_y = bin_smooth_data(x, np.array(y), bandwidth)
    kernel_smooth_y = kernel_smooth_data(x, np.array(y), bandwidth)
    lin_local_smooth_y = local_smooth(x, np.array(y), bandwidth, 0.5, 1)
    quad_local_smooth_y = local_smooth(x, np.array(y), bandwidth, 0.5, 2)


    plt.figure(figsize=(10, 6))
    plt.plot(x, y, color='black', label=f'{ticker} Closing Prices')
    plt.plot(x, bin_smooth_y, color='red', label='Bin Smoothing Fit')
    plt.plot(x, bin_smooth_y, color='orange', label='Kernel Smoothing Fit')
    plt.plot(x, bin_smooth_y, color='green', label='Local Linear Smoothing Fit')
    plt.plot(x, bin_smooth_y, color='blue', label='Local Quadratic Smoothing Fit')

    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.title(f"{ticker} Stock Price Over Time")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig("data.png")
    '''

# bin smooth for time series
def bin_smooth_data(xi, yi, bandwidth):
    """
    Smooth data using bin smoothing with a specified bandwidth.
    Parameters:
        xi (array): the x-coordinates of the data points (timestamps).
        yi (array): the y-coordinates of the data points (values to smooth).
        bandwidth (timedelta): the bandwidth (window size in x, in timedelta format).
    Returns:
        smoothed_xi (array): the smoothed x-coordinates.
        smoothed_yi (array): the smoothed y-coordinates.
    """
    smoothed_yi = []
    # convert bandwidth to days (or any other unit)
    bandwidth_in_days = bandwidth.days

    for i in range(len(xi)):
        x_in_window = []
        y_in_window = []

        # Find points within the bandwidth window
        for j in range(len(xi)):
            # calculate the time difference between xi[j] and xi[i]
            time_diff = (xi[j] - xi[i]).days  # Convert to days

            if abs(time_diff) <= bandwidth_in_days:
                x_in_window.append(xi[j])
                y_in_window.append(yi[j])

        # calculate the mean y value in the window
        mean_yi = np.mean(y_in_window) if y_in_window else np.nan
        smoothed_yi.append(mean_yi)
    
    return np.array(smoothed_yi)



# gaussian kernel function
def gaussian_kernel(x, bandwidth):
    return np.exp(-0.5 * (x / bandwidth) ** 2) / (bandwidth * np.sqrt(2 * np.pi))


# kernel smoothing function
def kernel_smooth_data(xi, yi, bandwidth):
    """
    Smooth data using kernel smoothing with a specified bandwidth.
    Parameters:
        xi (array): the x-coordinates of the data points.
        yi (array): the y-coordinates of the data points.
        bandwidth (float): the bandwidth (window size in x) for smoothing.
    Returns:
        smoothed_xi (array): the smoothed x-coordinates.
        smoothed_yi (array): the smoothed y-coordinates.
    """
    smoothed_yi = []
    # convert bandwidth to days (or any other unit)
    bandwidth_in_days = bandwidth.days

    for i in range(len(xi)):
        weights = []
        weighted_y = []

        # calculate weights for all points
        for j in range(len(xi)):

            time_diff = (xi[j] - xi[i]).days  
            weight = gaussian_kernel(time_diff, bandwidth_in_days)
            weights.append(weight)
            weighted_y.append(weight * yi[j])

        # calculate the smoothed value using weighted average
        smoothed_value = sum(weighted_y) / sum(weights)
        smoothed_yi.append(smoothed_value)

    return np.array(smoothed_yi)


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
    # convert bandwidth to days (or any other unit)
    bandwidth_in_days = bandwidth.days
    
    N = len(xi)
    num_neighbors = int(span * N)  # Number of neighbors to use in the local fit
    
    for i in range(N):

        weights = []
        x_local_diff = []
        x_local = []
        y_local = []

        time_diff = (xi - xi[i]).days  
        # get neighbors based on subinterval spa]
        distances = np.abs(time_diff)
        sorted_indices = np.argsort(distances)
        nearest_indices = sorted_indices[:num_neighbors]

        # get local (subinterval) data points
        x_local = xi[nearest_indices]
        y_local = yi[nearest_indices]

        # center x_local around xi[i]
        x_local_diff = (x_local - xi[i]).days

        # compute weights using triangular kernel
        u = x_local_diff / bandwidth_in_days
        for j in range(len(u)):
            weights.append(tri_weight(u[j]))

        # construct design matrix for the local regression (based on degree)
        M_local = create_M(np.array(x_local_diff), degree)
        
        # solve wls to find the coefficients of linear or quadratic polynomial (betas)
        c_local = weighted_least_squares(M_local, np.array(y_local), np.array(weights))

        # polynomial evaluated at xj = xi gives smoothed value (just constant term)
        y_smoothed = 0
        for k in range(len(c_local)):
            y_smoothed += c_local[k] * ((xi[i] - xi[i]).days)**k  

        smoothed_yi.append(y_smoothed)
    
    return np.array(smoothed_yi)

driver()