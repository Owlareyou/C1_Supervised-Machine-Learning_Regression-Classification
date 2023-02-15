# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 14:36:08 2023

@author: Gnij
"""

import numpy as np
import matplotlib.pyplot as plt

#file .mplstyle mplstyle is a Python package, which allows matplotlib users to simplify the process of improving plots' quality. Quite often font, size, legend, colors and other settings should be changed for making plots look better.
plt.style.use('./deeplearning.mplstyle')

x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

print(f"x_train = {x_train}")
print(f"y_train = {y_train}")


print(f"x_train.shape: {x_train.shape}")


#%%
#m = number of training
m = x_train.shape[0]    #ie. np.array.shape

print(f"number of training: {m}")


#aka len function
m2 = len(x_train)
print(f"number of training: is also {m2}")

#can be written like this
print(f"number of training: also also {len(x_train)}")




#%% this is how you perform a block of code
"""
training example i'th number
"""
i = 0 # ith number

x_i = x_train[i]
y_i = y_train[i]

print(f"the {i} th set of number is {x_i} and {y_i}")




#%%
"""
data plotting
"""

# scatter() function is matplotlib
# marker = 'x' and color = 'r'

plt.scatter(x_train,y_train, marker='x', c='r')

#set title
plt.title("Housing Prices")
#y axis
plt.ylabel('Price (1000)')
#x axis
plt.xlabel("Size (1000sqfft)")
plt.show()



#%%
"""
Model Function: x' = x*w +b
"""
w = 180
b = 100




def compute_model_output(x,w,b):
    """
    Computes the prediction of a linear model.
    Parameters
    ----------
    x : (ndarray (m,))
        data, m examples
    w,b : (scalar)
        model parameters.


    Returns
    -------
    y : (ndarray (m,y))
        data, m numbers of target values

    """

    m = x.shape[0]
    y = np.zeros(m)

    for i in range(m):
        y[i] = x[i] * w + b
    
    f_wb = y
    return f_wb




tmp_f_wb = compute_model_output(x_train, w, b)
#print(f"new number = {tmp_f_wb}")


#plot our model prediction
plt.plot(x_train, tmp_f_wb, c ='b', label = 'Our Prediction')


plt.scatter(x_train, y_train, marker='x', c ='r', label = 'Actual Value')

#set title
plt.title("Housing Prices")
#y axis
plt.ylabel('Price (1000)')
#x axis
plt.xlabel("Size (1000sqfft)")

plt.legend() #otherwise labels wont show
plt.show()








