# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 00:30:40 2023

@author: Gnij
"""

"""
Gradient Descent
"""
import math, copy
import numpy as np
from matplotlib import pyplot as plt

plt.style.use('./deeplearning.mplstyle')
#from lab_utils_uni import plt_house_x, plt_contour_wgrad, plt_divergence, plt_gradients

#%%
#np.array is more flexible than array.array and it can do specific calculations bwtween arrays
#load sample dataset
x_train = np.array([1.0, 2.0]) # features
y_train = np.array([300.0, 500.0])# targets



#%%%
#cost function
def compute_cost(x, y, w, b):
    m = x.shape[0]
    cost = 0
    
    for i in range(m):
        newcost = x[i] * w + b
        cost = cost + (newcost-y[i])**2
    
    total_cost = cost / (2*m)


    return total_cost
#%%
#compute gradient function
def compute_gradient(x, y, w, b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = x[i] * w + b
        dj_dw_tmp = (f_wb - y[i])*x[i]
        dj_db_tmp = (f_wb - y[i])
        dj_dw = dj_dw + dj_dw_tmp
        dj_db = dj_db + dj_db_tmp
    
        
        
    dj_dw = dj_dw * 1/m
    dj_db = dj_db * 1/m
    
    return dj_dw, dj_db

    
    
    


#%%
#gradient descent function
def gradient_descent(x, y, w_in, b_in, alpha, num_iters, cost_function, gradient_function):
    w = w_in
    b = b_in
    cost_history = []
    p_history = []
    #predict_history = []

    for iteration in range(num_iters):
        total_cost_eachiter = cost_function(x, y, w, b)
        
        #if total_cost < 10:
        #    return
        
        dj_dw, dj_db = gradient_function(x, y, w, b)
        
        w = w - alpha* dj_dw
        b = b - alpha* dj_db
    
        #prevent non-stop loop
        if iteration < 10000:
            cost_history.append(total_cost_eachiter)
            p_history.append([w,b])
            #predict_history.append()
        

        #print cost
        if iteration% math.ceil(num_iters/10) == 0:
            print(f"Iteration {iteration:4}: Cost {cost_history[-1]:0.2e} ",
                  '\n',
                  f"dj_dw: {dj_dw: 0.3e}, dj_db: {dj_db: 0.3e}  ",
                  '\n',
                  f"w: {w: 0.3e}, b:{b: 0.5e}",
                  '\n')



    return w, b, cost_history, p_history


#%%
# initialize parameters
w_init = 0
b_init = 0
# some gradient descent settings
iterations = 10000
tmp_alpha = 1.0e-2
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
print(f"(w,b) found by gradient descent: ({w_final:8.4f},{b_final:8.4f})")


#%%
#plot; not my code
# plot cost versus iteration  
#learn plt plotting
fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12,4))
ax1.plot(J_hist[:100])
ax2.plot(1000 + np.arange(len(J_hist[1000:])), J_hist[1000:])
ax1.set_title("Cost vs. iteration(start)");  ax2.set_title("Cost vs. iteration (end)")
ax1.set_ylabel('Cost')            ;  ax2.set_ylabel('Cost') 
ax1.set_xlabel('iteration step')  ;  ax2.set_xlabel('iteration step') 
plt.show()

#%%
#prediction time
print(f"1000 sqft house prediction {w_final*1.0 + b_final:0.1f} Thousand dollars")
print(f"1200 sqft house prediction {w_final*1.2 + b_final:0.1f} Thousand dollars")
print(f"2000 sqft house prediction {w_final*2.0 + b_final:0.1f} Thousand dollars")

#%%
#only in jupyter
#fig, ax = plt.subplots(1,1, figsize=(12, 6))
#plt_contour_wgrad(x_train, y_train, p_hist, ax)
#%%
"""
# initialize parameters
w_init = 0
b_init = 0
# set alpha to a large value
iterations = 10
tmp_alpha = 8.0e-1
# run gradient descent
w_final, b_final, J_hist, p_hist = gradient_descent(x_train ,y_train, w_init, b_init, tmp_alpha, 
                                                    iterations, compute_cost, compute_gradient)
"""