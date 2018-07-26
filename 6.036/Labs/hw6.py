import numpy as np
from hw1 import *
from math import factorial as fact
import math

##1)MAKING A PREDICTION

X = np.array([[1, 2, -1, 0],
              [3, -2, 1, 4]])
Y = np.array([[-1, 1, 1, 1]])

## Kernel function: x and z are col vectors
kernel_xz = lambda x,z : np.dot((x+1).T, z+1)
alpha = np.array([[1, 2, 0 , 0]])

def predict(X, Y, alpha, x, k):
    kernel_x = np.apply_along_axis(k, 0, X, x) + 1 #1Xn array
    result = np.sum(np.multiply\
                    (np.multiply(Y, alpha), kernel_x))
    return np.sign(result)

print (predict(X, Y, alpha, np.array([[1, -3]]).T, kernel_xz))

##2) COLONEL KERNEL 

#From the definition of the kernel function. phi and dimensions of x
#do the dot product of phi_x and phi_z. Then solve the given kernel. Result
#from pattern matching 
#are given

##c = [0, 1, ,1, 2, 8**0.5, 2, 0, 0, 0]

##3) CLASSIFICATION WITH 3 POINTS

#same as ##2)
#[((1, 1), 1), ((1, 0), -1), ((1, 1), 1)].

data = [(np.array([[1, 1]]).T, 1), (np.array([[1, 0]]).T, -1),\
        (np.array([[1, 1]]).T, 1)]
print (perceptron(data, 0)) # [2, 3, 0]

##4)KERNELS OF WISDOM

##4.1) Myster Kernel

##4.2) Polynomial Kernel

## second order polynomial kernel = (dot product of col vectors + 1)**2
## the length of a kth- order polynomial kernel transfomed feature vector
## D = Combination(d+k, k) where
## d is the original feature space and k is the order of the polynomial kernel

## for k = 2 and d = 1000
nCr = lambda n, r: fact(n)/(fact(r) * fact(n-r))
print (nCr(1002, 2)) #501501


##computing 2nd order polynomial kernel O(d^2)??

##confusion.......word documents to feature vectors 






    
                 
