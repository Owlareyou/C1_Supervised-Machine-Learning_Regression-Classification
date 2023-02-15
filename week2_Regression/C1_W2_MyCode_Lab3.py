# -*- coding: utf-8 -*-
"""
Created on Thu Feb  9 15:59:35 2023

@author: Gnij
"""

import numpy as np
import matplotlib.pyplot as plt
from lab_utils_multi import  run_gradient_descent 
from lab_utils_multi import  norm_plot, plt_equal_scale, plot_cost_i_w
from lab_utils_common import dlc
np.set_printoptions(precision=2)
plt.style.use('./deeplearning.mplstyle')

#%%
def load_house_data():
    data = np.loadtxt("./data/houses.txt", delimiter=',', skiprows=1)
    #print(data)
    X = data[:,:4]  #all of the houses, first four columns are features
    y = data[:,4]   #all of the houses, last column is our target 
    return X, y

X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']

fig, myplot = plt.subplots(1,4, figsize= (12,4), sharey= True)
for i in range(len(myplot)):
    myplot[i].scatter(X_train[:,i],y_train)
    myplot[i].set_xlabel(X_features[i])
    #print(i)
plt.show()



#%%
#very big alpha = does not converge

_,_, hist = run_gradient_descent(X_train, y_train, 100, alpha = 1e-7)
plot_cost_i_w(X_train, y_train, hist)

#%%
#zscore normalization:
#note: remember to output mu(training mean) and sigma(training std deviaiton) 
#       for future prediction

def zscore_normalize_feature(X):
    """
    Comput X, score normalize by column

    Parameters
    ----------
    X : ndarray (m,n)
        m houses and n features; input data

    Returns
    -------
    X_norm : ndarray (m,n)
            inpur normalizaed by column
    
    mu:     ndarray (n, )
            mean of each column, aka feature
    
    sigma:  ndarray (n, )
            standard devciation of each feature

    """    
    mu = np.mean(X, axis=0) # shape = (n,) #axis is important
    
    sigma = np.std(X, axis=0) #shape = (n,)
    
    X_norm = (X - mu)/sigma
    
    return X_norm, mu, sigma

#%%
#visualize feature scaling
mu     = np.mean(X_train,axis=0)   
sigma  = np.std(X_train,axis=0) 
X_mean = (X_train - mu)
X_norm = (X_train - mu)/sigma      

fig,ax=plt.subplots(1, 3, figsize=(12, 3))
ax[0].scatter(X_train[:,0], X_train[:,3])
ax[0].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[0].set_title("unnormalized")
ax[0].axis('equal')

ax[1].scatter(X_mean[:,0], X_mean[:,3])
ax[1].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[1].set_title(r"X - $\mu$")
ax[1].axis('equal')

ax[2].scatter(X_norm[:,0], X_norm[:,3])
ax[2].set_xlabel(X_features[0]); ax[0].set_ylabel(X_features[3]);
ax[2].set_title(r"Z-score normalized")
ax[2].axis('equal')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
fig.suptitle("distribution of features before, during, after normalization")
plt.show()

#%%
X_norm, X_mu, X_sigma = zscore_normalize_feature(X_train)
print(f"X_mu = {X_mu}, \nX_sigma = {X_sigma}")
print(f"Peak to Peak range by column in Raw        X:{np.ptp(X_train,axis=0)}")   
print(f"Peak to Peak range by column in Normalized X:{np.ptp(X_norm,axis=0)}")

#%%
fig,ax=plt.subplots(1, 4, figsize=(12, 3))
for i in range(len(ax)):
    norm_plot(ax[i],X_train[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count");
fig.suptitle("distribution of features before normalization")
plt.show()
fig,ax=plt.subplots(1,4,figsize=(12,3))
for i in range(len(ax)):
    norm_plot(ax[i],X_norm[:,i],)
    ax[i].set_xlabel(X_features[i])
ax[0].set_ylabel("count"); 
fig.suptitle("distribution of features after normalization")

plt.show()


#%%
w_norm, b_norm, hist = run_gradient_descent(X_norm, y_train, 1000, 1.0e-1, )

#%%
#predict target using normalized features
m = X_norm.shape[0]
yp = np.zeros(m)
for i in range(m):
    yp[i] = np.dot(X_norm[i], w_norm) + b_norm

    # plot predictions and targets versus original features    
fig,ax=plt.subplots(1,4,figsize=(12, 3),sharey=True)
for i in range(len(ax)):
    ax[i].scatter(X_train[:,i],y_train, label = 'target')
    ax[i].set_xlabel(X_features[i])
    ax[i].scatter(X_train[:,i],yp,color=dlc["dlorange"], label = 'predict')
ax[0].set_ylabel("Price"); ax[0].legend();
fig.suptitle("target versus prediction using z-score normalized model")
plt.show()


#%%
# First, normalize out example.
x_house = np.array([1200, 3, 1, 40])
x_house_norm = (x_house - X_mu) / X_sigma
print(x_house_norm)
x_house_predict = np.dot(x_house_norm, w_norm) + b_norm
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.0f}")

plt_equal_scale(X_train, X_norm, y_train)