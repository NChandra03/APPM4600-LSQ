import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime
from datetime import timedelta
from solver_noise import create_M, householder_qr, back_substitution
from WLSQ_working import weighted_least_squares
from data_smoothing import bin_smooth_data, kernel_smooth_data, local_smooth, gaussian_kernel, tri_weight

def driver():
    # Apple stock
    ticker = "AAPL"  
    start_date = "2020-01-01"
    end_date = "2023-12-01"

    # Download stock price data
    data = yf.download(ticker, start=start_date, end=end_date)

    # Extract x and y values
    x = data.index  # Dates / timestamps
    y = data['Close']  # Closing prices (stock price recorded at 4 pm ET)

    # Bandwidths: 30 and 90 days
    bandwidths = [timedelta(days=30), timedelta(days=90)]

    # Smoothing methods and LSR functions
    smoothing_methods = [
        ("Bin Smoothing", bin_smooth_data),
        ("Kernel Smoothing", kernel_smooth_data),
        ("Local Linear Smoothing", lambda xi, yi, bw: local_smooth(xi, yi, bw, 0.5, 1)),
        ("Local Quadratic Smoothing", lambda xi, yi, bw: local_smooth(xi, yi, bw, 0.5, 2)),
    ]

    # convert date strings to datetime objects
    start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
    end_datetime = datetime.strptime(end_date, "%Y-%m-%d")

    # convert datetime objects to timestamps
    start_timestamp = pd.to_datetime(start_datetime).timestamp()
    end_timestamp = pd.to_datetime(end_datetime).timestamp()
    
    n = 5
    N = len(y) 
    # linspace of timestamps
    timestamps = np.linspace(start_timestamp, end_timestamp, N)
    # convert timestamps back to datetime objects
    xeval = pd.to_datetime(timestamps, unit='s')
    # convert to numerical values for LSR (days since start)
    x_numeric = (xeval - xeval[0]).total_seconds() / (60 * 60 * 24)  # convert to days


    fig, axes = plt.subplots(4, 2, figsize=(16, 16), sharex=True, sharey=True)

    # Iterate over smoothing methods and bandwidths
    for row, (title, method) in enumerate(smoothing_methods):
        for col, bandwidth in enumerate(bandwidths):
            # Compute smoothed data
            smoothed_y = method(x, np.array(y), bandwidth)

            M = create_M(x_numeric, n)
            Q, R = householder_qr(M)
            Qt = np.transpose(Q)
            
            # LSR to original data (use xeval and original y)
            y_prime1 = Qt @ y
            c1 = back_substitution(np.array(R), np.array(y_prime1))
            c1 = np.array(c1)
            y_poly_original = sum(c1[i] * x_numeric ** i for i in range(n+1))  # Polynomial approximation to original data

            # LSR to smoothed data (use xeval and smoothed y)
            y_prime2 = Qt @ smoothed_y
            c2 = back_substitution(np.array(R), np.array(y_prime2))
            c2 = np.array(c2)
            y_poly_smooth = sum(c2[i] * x_numeric ** i for i in range(n+1))  # Polynomial approximation to smoothed data

            # plot data and smoothed fit
            axes[row, col].scatter(x, y, color='black', label=f'{ticker} Closing Prices', alpha=0.25)
            axes[row, col].scatter(x, smoothed_y, color='grey', label=f'{title} Data', alpha=0.25)
            axes[row, col].plot(xeval, y_poly_original, color='red', label=f'LSR to Original Data')
            axes[row, col].plot(xeval, y_poly_smooth, color='blue', label=f'LSR to Smoothed Data')

            axes[row, col].set_title(f"{title} (Bandwidth {bandwidth.days} Days)")
            axes[row, col].grid()
            if row == 3:  # Add x-axis label to the last row
                axes[row, col].set_xlabel("Time")
            if col == 0:  # Add y-axis label to the first column
                axes[row, col].set_ylabel("Stock Price")
            axes[row, col].legend()

    fig.tight_layout()
    fig.savefig("data_smoothed_compare_lsr_bandwidth.png")
    plt.close(fig)

    # separate figure overlaying all smoothing methods
    fig_overlay, ax = plt.subplots(figsize=(12, 8))
    colors = ['red', 'orange', 'green', 'blue']

    plt.plot(x, y, color='black', label=f'{ticker} Closing Prices', alpha=0.5)
   for method, title, color in zip(smoothing_methods, ['Bin Smoothing', 'Kernel Smoothing', 'Local Linear Smoothing', 'Local Quadratic Smoothing'], colors):
    smoothed_y = method[1](x, np.array(y), bandwidths[0])
    plt.plot(x, smoothed_y, label=f'{title} (BW {bandwidths[0].days} Days)', linestyle='--', color=color)

    plt.title(f'{ticker} Stock Price with All Smoothing Methods')
    plt.xlabel("Time")
    plt.ylabel("Stock Price")
    plt.legend()
    plt.grid()
    plt.savefig("data_smoothed_compare_overlay_lsr.png")
    plt.close(fig_overlay)

# Run the driver function
driver()
