# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 14:09:45 2023

@author: Gnij
"""

import numpy as np
import time


#%%
#vectpr creatopm

a = np.zeros(4)
print(f"np.zeros(4): \na = {a}, a shape = {a.shape}, a data type = {a.dtype}")

b = np.zeros((4,))
print(f"np.zeros((4,)): \nb = {b}, b shape = {b.shape}, a data type = {b.dtype}")

c = np.random.random_sample(4)
print(f"np.random.random_sample(4): \nc = {c}, c shape = {c.shape}, a data type = {c.dtype}")

#%%
#more data creation
a = np.arange(4.)
print(f"np.arange(4.): \na = {a}, a shape = {a.shape}, a data type = {a.dtype}")

c = np.random.rand(4)
print(f"np.random.rand(4): \nc = {c}, c shape = {c.shape}, a data type = {c.dtype}")

#%%
#can also be manially addressed
#%%
#operations on vectors: Indexing, Slicing
#indexing: refer to an element of an array by its position
#slicing means getting a subset of elements from an array based on their indeces

array_example = np.arange(10)

#access
print(f"array_example[2].shape: {array_example[2].shape}, array_example[2]: {array_example[2]}, \
      Accessing an element returns a scalar \n")
      
print(f"array_example[-1]: {array_example[-1]}")

try:
    c = array_example[10]
    
except Exception as e:
    print("\ntried to print variable c")
    print(f"The error is: {e}")
    
#%%
#slicing
a = np.arange(10)
print(f"a       = {a}")

c = a[2:7:1]; print(f"a[2:7:1]= {c}")
c = a[2:7:2]; print(f"a[2:7:2]= {c}")
c = a[3:]; print(f"a[3:]    = {c}")
c = a[:3]; print(f"a[:3]    = {c}")
c = a[:]; print(f"a[:]     = {c}")

#%%
#single vector opertations
a = np.arange(1,5)
b = -a
b = np.sum(a)
b = np.mean(a)
b = a**2
    
    
#%%
#vector vector element-wise operations
a = np.arange(1,5)
b = np.arange(1,5)
c = np.arange(1,3)
print(a+b)

try:
    print(a+c)
except Exception as e:
    print("the error that happened in this poerationis")
    print(e)

#%%
# show common Course 1 example
X = np.array([[1],[2],[3],[4]])
w = np.array([2])
c = np.dot(X[1], w)

print(f"X[1] has shape {X[1].shape}")
print(f"w has shape {w.shape}")
print(f"c has shape {c.shape}")

#%%
#matrix creation
a = np.zeros((5,5))
print(f"a shape = {a.shape}, \na = {a}")


a = np.zeros((1,3))
print(f"a shape = {a.shape}, \na = \n{a}")

a = np.random.random_sample((1,1))
print(f"a shape = {a.shape}, \na = {a}")

#%%
# NumPy routines which allocate memory and fill with user specified values
a = np.array([[5,1], [4,1], [3,1]]);   print(f" a shape = {a.shape}, np.array: a = {a}")
a = np.array([[5],   # One can also
              [4],   # separate values
              [3]]); #into separate rows
print(f" a shape = {a.shape}, np.array: a = {a}")

#%%
a = np.arange(6).reshape(-1,2)
print(f"a.shape={a.shape}")

#access an element
print(f"\na[2,0].shape:   {a[2, 0].shape}, a[2,0] = {a[2, 0]},     type(a[2,0]) = {type(a[2, 0])} Accessing an element returns a scalar\n")

#access a row
print(f"a[2].shape:   {a[2].shape}, a[2]   = {a[2]}, type(a[2])   = {type(a[2])}")

#%%
#vector 2-D slicing operations
a = np.arange(20).reshape(-1, 10)
print(f"a = \n{a}")

#access 5 consecutive elements (start:stop:step)
print("a[0, 2:7:1] = ", a[0, 2:7:1], ",  a[0, 2:7:1].shape =", a[0, 2:7:1].shape, "a 1-D array")

#access 5 consecutive elements (start:stop:step) in two rows
print("a[:, 2:7:1] = \n", a[:, 2:7:1], ",  a[:, 2:7:1].shape =", a[:, 2:7:1].shape, "a 2-D array")

# access all elements
print("a[:,:] = \n", a[:,:], ",  a[:,:].shape =", a[:,:].shape)

# access all elements in one row (very common usage)
print("a[1,:] = ", a[1,:], ",  a[1,:].shape =", a[1,:].shape, "a 1-D array")
# same as
print("a[1]   = ", a[1],   ",  a[1].shape   =", a[1].shape, "a 1-D array")


























