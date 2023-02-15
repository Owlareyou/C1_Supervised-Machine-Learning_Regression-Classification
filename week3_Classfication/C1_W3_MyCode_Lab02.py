# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 22:29:56 2023

@author: jchua
"""

import numpy as np
import matplotlib.pyplot as plt
from plt_one_addpt_onclick import plt_one_addpt_onclick
from lab_utils_common import draw_vthresh
plt.style.use('./deeplearning,.mplstyle')

#%%

input_array = np.array([1,2,3])
exp_array = np.exp(input_array)


print("Input to exp:", input_array)
print("Output of exp:", exp_array)


input_val = 1  
exp_val = np.exp(input_val)

print("Input to exp:", input_val)
print("Output of exp:", exp_val)
#%%

def sigmoid(z):
    sigmoid = 1 / (1+np.exp(z))
    
    return sigmoid
#%%
# Generate an array of evenly spaced values between -10 and 10
z_tmp = np.arange(-10,11)

# Use the function implemented above to get the sigmoid values
y = sigmoid(z_tmp)

# Code for pretty printing the two arrays next to each other
np.set_printoptions(precision=3) 
print("Input (z), Output (sigmoid(z))")
print(np.c_[z_tmp, y])

#%%
# Plot z vs sigmoid(z)
fig,ax = plt.subplots(1,1,figsize=(5,3))
ax.plot(z_tmp, y, c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
