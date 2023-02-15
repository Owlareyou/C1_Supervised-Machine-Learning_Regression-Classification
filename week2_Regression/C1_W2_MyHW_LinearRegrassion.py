# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 15:02:56 2023

@author: jchua
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import *
import copy
import math
from public_tests import *
np.set_printoptions(precision=2)


def load_data():
    data = np.loadtxt("data/ex1data1.txt", delimiter=',')
    X = data[:,0]
    y = data[:,1]
    return X, y
# load the dataset
x_train, y_train = load_data()
#%%
print(x_train[: 5]) #population of  city * 10,000
print(y_train[: 5]) #restaurant profit * 10,000
print(type(x_train))
print(type(y_train))
print(x_train.shape)
print(y_train.shape)

plt.scatter(x_train, y_train, c = 'r')

plt.title("profit vs population per city")
plt.xlabel("population")
plt.ylabel("profit")
plt.show()
#%%
# UNQ_C1
# GRADED FUNCTION: compute_cost
#w_init = 2
#b_init = 1


def compute_cost(x, y, w, b): 
    """
    Computes the cost function for linear regression.
    
    Args:
        x (ndarray): Shape (m,) Input to the model (Population of cities) 
        y (ndarray): Shape (m,) Label (Actual profits for the cities)
        w, b (scalar): Parameters of the model
    
    Returns
        total_cost (float): The cost of using w,b as the parameters for linear regression
               to fit the data points in x and y
    """
    # number of training examples
    m = x.shape[0] 
    total_cost = 0    
    
    #prediction = w*x + b
    prediction = np.multiply(x,w) +b 

    difference = np.subtract(prediction, y)
    powerof2 = np.square(difference)
    total_cost = (1/(2*m)) * np.sum(powerof2)
    

    return total_cost
"""
cost = compute_cost(x_train, y_train, w_init, b_init)
print(cost)
"""
#%%

def compute_gradient(x,y,w,b):

    m = x.shape[0] 
    
    
    #prediction = w*x + b
    prediction = np.multiply(x,w) +b 

    difference = np.subtract(prediction, y)
    djdb = (1/(m)) * np.sum(difference)
    djdw = (1/(m)) * np.sum(np.multiply(difference,x))


    return djdw, djdb
"""
# Compute and display gradient with w initialized to zeroes
initial_w = 0
initial_b = 0

tmp_dj_dw, tmp_dj_db = compute_gradient(x_train, y_train, initial_w, initial_b)
print('Gradient at initial w, b (zeros):', tmp_dj_dw, tmp_dj_db)
"""


#%%
def gradient_descent (x,y,w_in,b_in,alpha,compute_cost,compute_gradient,num_iter):
    """
    w = w - alpha djdw
    b = b - alpha djdb
    """
    cost_hist = []
    w = w_in
    b = w_in

    
    for i in range(num_iter):
        dj_dw, dj_db = compute_gradient(x, y, w, b)    
        
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
   
        cost = compute_cost(x, y, w, b)
        cost_hist.append(cost)
    
    
    
    
    
    
    return w, b, cost_hist
    
w_in = 0.
b_in = 0.
alpha = 0.01
num_iter = 1500

final_w, final_b, cost_hist = gradient_descent(x_train, y_train, w_in, b_in, alpha, compute_cost, compute_gradient, num_iter)

print(final_w, final_b)


final_prediction = np.multiply(x_train,final_w) + final_b

#linear
plt.plot(x_train, final_prediction, c = 'b')

plt.scatter(x_train, y_train, c='r')
# Set the title
plt.title("Profits vs. Population per city")
# Set the y-axis label
plt.ylabel('Profit in $10,000')
# Set the x-axis label
plt.xlabel('Population of City in 10,000s')
plt.show()

#%%
predict1 = 3.5 * final_w + final_b
print('For population = 35,000, we predict a profit of $%.2f' % (predict1*10000))

predict2 = 7.0 * final_w+ final_b
print('For population = 70,000, we predict a profit of $%.2f' % (predict2*10000))

