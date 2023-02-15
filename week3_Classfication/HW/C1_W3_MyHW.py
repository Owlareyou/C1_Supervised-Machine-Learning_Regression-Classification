# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 23:56:11 2023

@author: jchua
"""

import numpy as np
import matplotlib.pyplot as plt
#from utils import plot_data # you have to compile utils.py first
from utils import *

import copy
import math

#%%
filename = str("data/ex2data1.txt") #remember where is your location
data = np.loadtxt(filename, delimiter=',')

X_train = data[:,:2]
y_train = data[:,-1]



#%%
print("First five elements in X_train are:\n", X_train[:5])
print("Type of X_train:",type(X_train))

print("First five elements in y_train are:\n", y_train[:5])
print("Type of y_train:",type(y_train))


#%%
# Plot examples
plot_data(X_train, y_train[:], pos_label="Admitted", neg_label="Not admitted")

# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()

#%%
def sigmoid(z):
    sig = 1 / (1+np.exp(-z))    
    

    return sig

#%%
def compute_cost(X, y, w, b, *argv):
    m = X.shape[0]
    
    #takealook = np.sum(np.multiply(X,w),1)
    z = np.sum(np.multiply(X,w),1) + b
    f_wb = sigmoid(z)    
    
    #tmp = np.log(f_wb)
    #tmpp = -y
    loss_1 =  (-y)*np.log(f_wb)
    loss_2 = (1-y)*np.log(1-f_wb)
    loss = loss_1 - loss_2
    
    j_wb = 1/m * np.sum(loss)
#    1/m sigmoid (loss())
    
    
    return j_wb   

#%%
def compute_gradient(X, y, w, b, *argv): 
    
    m,n = X.shape
    djdw = []
    djdb = []
    tmp = []
    
    #takealook = np.sum(np.multiply(X,w),1)
    for i in range(n):
        
        z = np.sum(np.multiply(X,w),axis = 1) + b
        f_wb = sigmoid(z)
        
        difference = np.subtract(f_wb, y)
        djdb = (1/(m)) * np.sum(difference)#no problem
        djdw = (1/(m)) * np.sum(np.multiply(difference,X[:,i]))  #no problem with math, problem with index
        tmp = np.append(tmp,djdw)#was list, need array
    
    
    return djdb, tmp #there are two w's

    
#%%
# Compute and display gradient with w and b initialized to zeros
m,n = X_train.shape
initial_w = np.zeros(n)
initial_b = 0.

dj_db, dj_dw = compute_gradient(X_train, y_train, initial_w, initial_b)
print(f'dj_db at initial w and b (zeros):{dj_db}' )
print(f'dj_dw at initial w and b (zeros):{dj_dw}' )


#%%
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters, lambda_): 
    """
    Performs batch gradient descent to learn theta. Updates theta by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X :    (ndarray Shape (m, n) data, m examples by n features
      y :    (ndarray Shape (m,))  target value 
      w_in : (ndarray Shape (n,))  Initial values of parameters of the model
      b_in : (scalar)              Initial value of parameter of the model
      cost_function :              function to compute cost
      gradient_function :          function to compute gradient
      alpha : (float)              Learning rate
      num_iters : (int)            number of iterations to run gradient descent
      lambda_ : (scalar, float)    regularization constant
      
    Returns:
      w : (ndarray Shape (n,)) Updated values of parameters of the model after
          running gradient descent
      b : (scalar)                Updated value of parameter of the model after
          running gradient descent
    """
    
    # number of training examples
    m = len(X)
    
    # An array to store cost J and w's at each iteration primarily for graphing later
    J_history = []
    w_history = []
    
    for i in range(num_iters):

        # Calculate the gradient and update the parameters
        dj_db, dj_dw = gradient_function(X, y, w_in, b_in, lambda_)   

        # Update Parameters using w, b, alpha and gradient
        w_in = w_in - alpha * dj_dw               
        b_in = b_in - alpha * dj_db              
       
        # Save cost J at each iteration
        if i<100000:      # prevent resource exhaustion 
            cost =  cost_function(X, y, w_in, b_in, lambda_)
            J_history.append(cost)

        # Print cost every at intervals 10 times or as many iterations if < 10
        if i% math.ceil(num_iters/10) == 0 or i == (num_iters-1):
            w_history.append(w_in)
            print(f"Iteration {i:4}: Cost {float(J_history[-1]):8.2f}   ")
        
    return w_in, b_in, J_history, w_history #return w and J,w history for graphing
#%%
np.random.seed(1)
initial_w = 0.01 * (np.random.rand(2) - 0.5)
initial_b = -8

# Some gradient descent settings
iterations = 10000
alpha = 0.001

w,b, J_history,_ = gradient_descent(X_train ,y_train, initial_w, initial_b, 
                                   compute_cost, compute_gradient, alpha, iterations, 0)
#%%
plot_decision_boundary(w, b, X_train, y_train)
# Set the y-axis label
plt.ylabel('Exam 2 score') 
# Set the x-axis label
plt.xlabel('Exam 1 score') 
plt.legend(loc="upper right")
plt.show()
#%%
def predict(X,w,b):
   
    z = np.sum(np.multiply(X,w),axis = 1) + b
    sig = sigmoid(z)
    #sig = 1 / (1+np.exp(-z)) 
    probability = sig
    probability = np.where(probability >= 0.5, 1, 0)
    
    return probability

#%%
# Test your predict code
np.random.seed(1)
tmp_w = np.random.randn(2)
tmp_b = 0.3    
tmp_X = np.random.randn(4, 2) - 0.5
tmp_p = predict(tmp_X, tmp_w, tmp_b)
print(f'Output of predict: shape {tmp_p.shape}, value {tmp_p}')

#Compute accuracy on our training set
p = predict(X_train, w,b)
print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))
#%%
# =============================================================================
# NEW PROBLEM AND DATASET
# 
# =============================================================================
#X_train, y_train = load_data("data/ex2data2.txt")
filename = str("data/ex2data2.txt")

data = np.loadtxt(filename, delimiter=',')
X_train = data[:,:2]
y_train = data[:,-1]

# print X_train
print("X_train:")
print(X_train[:5])
print("Type of X_train:")
print(type(X_train))
print("y_train:")
print( y_train[:5])
print("Type of y_train:")
print(type(y_train))

#%%
# Plot examples
plot_data(X_train, y_train[:], pos_label="Accepted", neg_label="Rejected")

# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()

#%%
#feature mapping: transform into 2-dimension vector, fit the data better.

print("Original shape of data:", X_train.shape)

mapped_X =  map_feature(X_train[:, 0], X_train[:, 1])
print("Shape after feature mapping:", mapped_X.shape)


print("X_train[0]:", X_train[0])
print("mapped X_train[0]:", mapped_X[0])
#%%
def compute_cost_reg(X, y, w, b, lambda_ = 1):
    m = X.shape[0]
    
    #takealook = np.sum(np.multiply(X,w),1)
    z = np.sum(np.multiply(X,w),1) + b
    f_wb = sigmoid(z)    
    
    regulerization_term = (lambda_ / (2*m)) *np.sum(np.power(w,2))
    
    #tmp = np.log(f_wb)
    #tmpp = -y
    loss_1 =  (-y)*np.log(f_wb)
    loss_2 = (1-y)*np.log(1-f_wb)
    loss = loss_1 - loss_2
    
    j_wb = 1/m * np.sum(loss)
#    1/m sigmoid (loss())

    j_wb = j_wb + regulerization_term
    
    return j_wb      
#%%
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1]) - 0.5
initial_b = 0.5
lambda_ = 0.5
cost = compute_cost_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print("Regularized cost :", cost)

# UNIT TEST    
#compute_cost_reg_test(compute_cost_reg)

#%%
def compute_gradient_reg(X, y, w, b, lambda_ = 1): 
    m,n = X.shape
    djdw = []
    djdb = []
    tmp = []
    
    #takealook = np.sum(np.multiply(X,w),1)
    for i in range(n):
        
        z = np.sum(np.multiply(X,w),axis = 1) + b
        f_wb = sigmoid(z)
        
        difference = np.subtract(f_wb, y)
        djdb = (1/(m)) * np.sum(difference)#no problem
        tmp = (1/(m)) * np.sum(np.multiply(difference,X[:,i]))  #no problem with math, problem with index
        djdw = np.append(djdw,tmp)#was list, need array
    
    regulerization_term = np.multiply((lambda_ / (m)),w)
    
    djdw = djdw + regulerization_term
    
    
    return djdb, djdw #there are two w's    

#%%
X_mapped = map_feature(X_train[:, 0], X_train[:, 1])
np.random.seed(1) 
initial_w  = np.random.rand(X_mapped.shape[1]) - 0.5 
initial_b = 0.5
 
lambda_ = 0.5
dj_db, dj_dw = compute_gradient_reg(X_mapped, y_train, initial_w, initial_b, lambda_)

print(f"dj_db: {dj_db}", )
print(f"First few elements of regularized dj_dw:\n {dj_dw[:4].tolist()}", )

# UNIT TESTS    
#compute_gradient_reg_test(compute_gradient_reg)
#%%
# Initialize fitting parameters
np.random.seed(1)
initial_w = np.random.rand(X_mapped.shape[1])-0.5
initial_b = 1.

# Set regularization parameter lambda_ (you can try varying this)
lambda_ = 0.01    

# Some gradient descent settings
iterations = 10000
alpha = 0.01

w,b, J_history,_ = gradient_descent(X_mapped, y_train, initial_w, initial_b, 
                                    compute_cost_reg, compute_gradient_reg, 
                                    alpha, iterations, lambda_)

#%%
plot_decision_boundary(w, b, X_mapped, y_train)
# Set the y-axis label
plt.ylabel('Microchip Test 2') 
# Set the x-axis label
plt.xlabel('Microchip Test 1') 
plt.legend(loc="upper right")
plt.show()

#%%
#Compute accuracy on the training set
p = predict(X_mapped, w, b)

print('Train Accuracy: %f'%(np.mean(p == y_train) * 100))









