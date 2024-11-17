# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 19:00:55 2024

@author: Zara Chandra
"""
import numpy as np
import matplotlib.pyplot as plt
import random

plt.close('all')

def uniform(n,minimum,maximum):
    error = []
    for i in range (n):
        error.append(random.random() * (maximum - minimum) + minimum)

    return(error)

def uniformstd(n,std_dev):
    maximum = std_dev * np.sqrt(12) * 1/2
    minimum = -std_dev * np.sqrt(12) * 1/2
    error = []
    for i in range (n):
        # centering noise at 0, range is  -(b - a)/2 to (b - a)/2 
        error.append(random.random() * (maximum - minimum) - (maximum - minimum) / 2)
        
    return(error)

def normal(n,sigma):
    mu = 0
    error = []
    for i in range (n):
        error.append(random.gauss(mu, sigma))

    return(error)

x1 = uniform(10000,-5,5)
x2 = uniformstd(10000,10/(np.sqrt(12)))
x3 = normal(10000,10/(np.sqrt(12)))
plt.figure(1)
plt.hist([x1,x2],bins = 10)
plt.figure(2)
plt.hist(x3,bins = 100)