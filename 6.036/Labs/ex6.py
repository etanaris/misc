import numpy as np
from hw1 import *


phi = lambda x: (x**(np.indices(x.shape)[1]+1))
x = np.array([[3,2,1]])
y = np.array([[5,1,2]])
print (x, phi(x), phi(y))#[3 4 1]
print (np.dot(phi(x), phi(y).T)) #27

#each alpha is a counter for how many times perceptron messed up on that point
# becasue if the inital starting point was the null vector, then your final
#theta is just the sum of y.x over all the missclassified points until it
#converges to  the first accurate classifier

#a) used the code in hw1.py to find the misses per point. can do this
#manually too.. [1, 0, 1]

data = [((np.array([[1,-1]])).T ,1),((np.array([[0,1]])).T, -1), \
         ((np.array([[-1.5,-1]])).T, 1)]
print (perceptron(data, 0)) #[1, 0, 1]

#b) same deal here. R is higher tho so perceptron takes longer to converge
# (R/gamma)**2

data_2 = [((np.array([[1,-1]])).T ,1),((np.array([[0,1]])).T, -1), \
         ((np.array([[-10,-1]])).T, 1)]

print (perceptron(data_2, 0)) #[5, 0 , 1]




