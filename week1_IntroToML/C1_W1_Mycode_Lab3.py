# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 23:57:27 2023

@author: Gnij
"""

#%%
"""
Cost Function
"""

import numpy as np
#%matplotlib widget
import matplotlib.pyplot as plt

#from lab_utils_uni import plt_intuition, plt_stationary, plt_update_onclick, soup_bowl

plt.style.use('./deeplearning.mplstyle')



x_train = np.array([1.0, 2.0])
y_train = np.array([300.0, 500.0])

w = 100 #cost min when w = 200
b = 100


def compute_cost(x, y, w, b):
    """
    Comput cost function for lienar regression

    Parameters
    ----------
    x : (ndarray (m,))
        x_train, data
    y : (ndarray (m,))
        y_train. target
    w : scalar
        model parameter.
    b : scalar
        model parameter.

    Returns
    -------
    total_cost: float
                the cost of using w, b as parameter in comparason to x and y

    """
    
    f_wb = np.zeros(x.shape)
    total_cost = 0
    #m = x.shape[0]
    
    for i in range(x.shape[0]):
        f_wb[i] = x[i] * w + b
    
    
    total_cost = np.sum(np.square(np.subtract(f_wb, y))) / (2*(x.shape[0]))
    
    return total_cost


total_cost = compute_cost(x_train, y_train, w, b)
print(f"this is the total cost: {total_cost}")
    
    
    
    
    