# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 00:31:22 2023

@author: Gnij
"""

import math, copy
import numpy as np
import matplotlib.pyplot as plt

plt.style.use("./deeplearning.mplstyle")

np.set_printoptions(precision=2)

#%%
#size, num. of bedroom, num. of floors, age of home, price(1000s)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

#display input data
print(X_train); print(X_train.shape)

#%%
#w is a 1-D vector; b will be a scalar parameter

b_init = 785.1811367994083
w_init = np.array([ 0.39133535, 18.75376741, -53.36032453, -26.42131618])
print(f"w_init shape: {w_init.shape}, b_init type: {type(b_init)}")



#model prediction(multi-variables)

#use dot product
def predict (x, w, b):
    """
    single predict using linear regression
    Args:
      x (ndarray): Shape (n,) example with multiple features
      w (ndarray): Shape (n,) model parameters   
      b (scalar):             model parameter 
      
    Returns:
      p (scalar):  prediction
    """
    predict = np.dot(x,w) + b
    return predict

# get a row from our training data
x_vec = X_train[0,:]
print(f"x_vec shape {x_vec.shape}, x_vec value: {x_vec}")

# make a prediction
f_wb = predict(x_vec,w_init, b_init)
print(f"f_wb shape {f_wb.shape}, prediction: {f_wb}")

#%%
#compute cost(multi-variables)

#pseudo code for cost
#total cost = 1/2m * add_all_together (predict value - real value)**2

def compute_cost (X,y,w,b):
    """
    compute cost
    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """
    m = X.shape[0]
    predict = np.dot(X,w) +b
    difference = np.subtract([predict], y)
    powerof2 = difference**2
    total_cost = (1/2*m) * np.sum(powerof2)
    
    return total_cost

cost = compute_cost(X_train, y_train, w_init, b_init)
print(cost)

#%%
def compute_gradient(X,y,w,b):
    """
    compute the gradient bewteen data and parameters

    Parameters
    ----------
    X : array(m,n)
        data.
    y : array(n)
        target.
    w : scalar
        parameter.
    b : scalar
        parameter.

    Returns
    -------
    dj_dw : array(1,n)
        partial differencialtion;The gradient of the cost w.r.t. the parameters w. 
    dj_db : scalar
        partial differncialtion;The gradient of the cost w.r.t. the parameter b. 

    """
    m = X.shape[0]
    predict = np.dot(X,w) +b 
    difference = np.subtract([predict], y_train)
    diff_for_djdb = difference
    difference = np.reshape(difference, (-1,1))
    difference = np.append(difference, difference, axis =1)
    difference = np.append(difference, difference, axis =1)
    #print(difference)

    timesx = difference * X_train
    #print(timesx)

    dj_dw = np.mean(timesx, axis=0)
    #print(dj_dw)

    dj_db = (1/m) * np.sum(diff_for_djdb)
    #print(dj_db.shape)
    
    return dj_dw, dj_db

#Compute and display gradient 
tmp_dj_db, tmp_dj_dw = compute_gradient(X_train, y_train, w_init, b_init)

print(f'dj_dw at initial w,b: \n {tmp_dj_dw}') # this is correct
print(f'dj_db at initial w,b: {tmp_dj_db}') # this is incorrect


#%%
def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    """
    Performs batch gradient descent to learn w and b. Updates w and b by taking 
    num_iters gradient steps with learning rate alpha
    
    Args:
      X (ndarray (m,n))   : Data, m examples with n features
      y (ndarray (m,))    : target values
      w_in (ndarray (n,)) : initial model parameters  
      b_in (scalar)       : initial model parameter
      cost_function       : function to compute cost
      gradient_function   : function to compute the gradient
      alpha (float)       : Learning rate
      num_iters (int)     : number of iterations to run gradient descent
      
    Returns:
      w (ndarray (n,)) : Updated values of parameters 
      b (scalar)       : Updated value of parameter 
      cost_hist        : History of previous costs
      """
    w = w_in
    b = b_in
    # a history array of cost
    cost_hist = []
    # a history array of w and b
    
    for iteration in range(num_iters):
        #compute gradient
        dj_dw, dj_db = gradient_function(X,y,w,b)
        
        #update w and b
        w = w - (alpha * dj_dw)
        b = b - (alpha * dj_db)
        
        current_cost = cost_function(X,y,w,b)
        cost_hist = np.append(cost_hist, current_cost)
        if iteration % 50 == 0:
            print(f"iteration: {iteration}")
            print(f"current cost: {current_cost}")
            
        
    
    return w, b, cost_hist 
#%%

# initialize parameters
initial_w = np.zeros_like(w_init)
initial_b = 0.
# some gradient descent settings
iterations = 1000
alpha = 5.0e-7
# run gradient descent 
w_final, b_final, J_hist = gradient_descent(X_train, y_train, initial_w, initial_b,
                                                    compute_cost, compute_gradient, 
                                                    alpha, iterations)
print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,_ = X_train.shape
for i in range(m):
    print(f"prediction: {np.dot(X_train[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")
    
    
#%%
# LEARN HOW TO PLOT
# plot cost versus iteration  


fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
ax1.plot(J_hist)
ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. iteration");  ax2.set_title("Cost vs. iteration (tail)")
ax1.set_ylabel('Cost')             ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')   ;  ax2.set_xlabel('iteration step') 
plt.show()
#%%
#my try on practice ploting
fig,(ax1, ax2)= plt.subplots(1,2, figsize=(12, 4))
#ax1 ax2 is the plot's name
ax1.plot(J_hist)                    ; ax2.plot(100 + np.arange(len(J_hist[100:])), J_hist[100:])
ax1.set_title("Cost vs. Iteration") ; ax2.set_title("Cost vs Iter (100~:)")
ax1.set_xlabel('cost')              ; ax2.set_xlabel('cost')
ax2.set_ylabel('iteration steps')   ; ax2.set_ylabel('iteration')


plt.show()
    
    
    
    
    
    