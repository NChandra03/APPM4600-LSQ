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

    # apple stock
    ticker = "AAPL"
    train_start_date = "2020-01-01"
    train_end_date = "2023-12-01"
    test_end_date = "2024-12-08"

    # download stock price data
    train_data = yf.download(ticker, start=train_start_date, end=train_end_date)
    test_data = yf.download(ticker, start=train_end_date, end=test_end_date)

    # combine training and testing data for consistent plotting
    full_data = pd.concat([train_data, test_data])

    # extract x and y values
    x_full = (full_data.index - full_data.index[0]).days.to_numpy()
    y_full = full_data['Close'].to_numpy()

    # training data for fitting
    x_train = x_full[:len(train_data)]
    y_train = y_full[:len(train_data)]

    # lsr order of approximation
    degrees = [2, 3, 4, 5]

    fig, axes = plt.subplots(len(degrees), 1, figsize=(12, len(degrees) * 4))
  
    for i, n in enumerate(degrees):

        M = create_M(x_train, n)
        Q, R = householder_qr(M)
        Qt = np.transpose(Q)

        # LSR approximation based on test time interval
        y_prime = Qt @ y_train
        c = back_substitution(np.array(R), np.array(y_prime))
        c = np.array(c)
        y_poly = sum(c[i] * x_full ** i for i in range(n+1))  

        axes[i].plot(train_data.index, y_train, color='blue', label='Training Data')
        axes[i].plot(test_data.index, y_full[len(train_data):], color='grey', label='Test Data', alpha=0.5)
        axes[i].plot(full_data.index, y_poly, color='red', label=f'Polynomial Fit (Degree {n})')
        axes[i].axvline(train_data.index[-1], color='black', linestyle='--', label='Train/Test Split')
        axes[i].set_title(f"Degree {n} Polynomial Fit")
        axes[i].set_xlabel("Date")
        axes[i].set_ylabel("Stock Price")
        axes[i].legend()
        axes[i].grid()

    plt.tight_layout()
    plt.savefig("data_test_interval.png", dpi=300)
    plt.show()
    plt.figure()
    

driver()
