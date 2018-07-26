import numpy as np
from lab1 import *

#calculation in notes

#1)Intro to linear regression
# Empirical risk : Average squared loss
#inverting a matrix takes close to O(n^3)

#2)lisa and andry

##100 training points: working in R2 vs R50


##estimation error(variance) results when you have a small noisy training set.

##structural error(bias) results when you try to apply a linear model on a non
##linear dataset.For instance, it might result from using some form of
##feature expansion and doing linear regression in a higher feature space.
##You're more likely to find a viable set of weight parameters. But
##here, you run the risk of infering a set of parameters that don't generalize
##well which increases your estimation error(variance. This is because you're
##working with an nXD matrix, where D>>d, which means n may be smaller than D
##(underdetermined case), or  n is still larger than D but not by much, which
## means you have few data points to predict the model with resonable
##certainity in the higher dimension. i.e. you landed with a small noisy
##training set in the higher feature space => higher estimation error)

##regularization makes the mean stray from the 'true' weight vector. So,by
##introducing bias it makes the variance smaller.


##3)Adding regularization


##4)Minimizing emprical risk 
##theta = np.dot(np.linalg.inv(np.dot(X,X.T)), np.dot(X,Y.T))

##5)Flashback to classification

# Write a function that returns the gradient of hinge(v) in terms of the arguments provided.
# assume values of v and dv as input.
def d_hinge(v, dv):
    return np.where(v>=1, 0, -dv)

# Write a function that returns the gradient of hinge_loss(x, y, th, th_0)
# with respect to th
def d_hinge_loss_th(x, y, th, th0):
    return d_hinge(hinge_loss(x, y, th, th0))

# Write a function that returns the gradient of hinge_loss(x, y, th, th_0)
# with respect to th0
def d_hinge_loss_th0(x, y, th, th0):
    return np.where(y*(np.dot(th.T, x) + th0)>=1, 0, -y*th0)

# Write a function that returns the gradient of svm_obj(x, y, th, th_0) with 
# respect to th
def d_svm_obj_th(x, y, th, th0, lam):
    return d_hinge_loss_th(x, y, th, th0) + 2*lam*np.linalg.norm(th)
    
def d_svm_obj_th0(x, y, th, th0, lam):
    return d_hinge_loss_th0(x,y,th, th0)
