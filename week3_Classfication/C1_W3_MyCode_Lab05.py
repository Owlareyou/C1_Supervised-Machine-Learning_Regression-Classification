# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:51:58 2023

@author: jchua
"""

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')
#%%
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)

fig,ax = plt.subplots(1,1,figsize=(4,4))
#plot_data(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()
#%%
def sigmoid(z):
    
    sig = 1/ (1+(np.exp(-z)))
    
    return sig

#%%

def compute_cost_logistic(X,y,w,b):

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
w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))

