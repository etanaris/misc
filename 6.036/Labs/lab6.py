import pdb
import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
import math
from hw6 import predict 

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

def cv(value_list):
    return np.transpose(rv(value_list))

######################################################################
# Kernels
######################################################################


# (x.T*z +1) ^n
def k_poly(x, z, n=1):
    return (np.dot(x.T, z) + 1)**n


def k_gauss(x, z, beta=1):
    return math.exp(-beta*(np.dot((x-z).T, x-z)))


######################################################################
# Kernel perceptron
######################################################################

def predict(X, Y, alpha, x, k):# remove the +1 if it's a
                                            #kerne perceptron without offset
    kernel_x = (np.apply_along_axis(k, 0, X, x) + 1)#1Xn array
    result = np.sum(np.multiply\
                    (np.multiply(Y, alpha), kernel_x))
    return result, np.sign(result)

def kernel_perceptron(X, Y, k, T =10):
    d, n = x.shape
    alphas = np.zeros((1,n))
    counter = 0
    while(counter < T):
        for i in range(n):
            counter+=1
            if Y[i]*predict(X, Y, alpha, X[:,[i]], k) <= 0:
                alphas[i]+=1
    return alphas

######################################################################
# Kernel (ridge) regression
######################################################################
#alpha's nX1
def predict_regression(X, alpha, x, k):
    return float(np.dot(np.apply_along_axis(k, 0, X, x), alpha.T))

def gram(X, k):
    d, n = X.shape
    return np.vstack((np.apply_along_axis(k, 0, X, X[:,[i]]) for i in range(n)))
            
            
def kernel_lin_reg(X, Y, lam, k):
    d, n = X.shape
    kernel_matrix = gram(X,k) + lam*np.eye(n) #nXn
    return np.dot(np.linalg.inv(kernel_matrix), Y.T)
    

######################################################################
# Data
######################################################################

# Used to define X2, Y2
def get_classify_data():
    X= np.array([[-0.23390341,  1.18151883, -2.46493986,  1.55322202,  1.27621763,
              2.39710997, -1.3440304 , -0.46903436, -0.64673502, -1.44029872,
              -1.37537243,  1.05994811, -0.93311512,  1.02735575, -0.84138778,
              -2.22585412, -0.42591102,  1.03561105,  0.91125595, -2.26550369],
             [-0.92254932, -1.1030963 , -2.41956036, -1.15509002, -1.04805327,
              0.08717325,  0.8184725 , -0.75171045,  0.60664705,  0.80410947,
              -0.11600488,  1.03747218, -0.67210575,  0.99944446, -0.65559838,
              -0.40744784, -0.58367642,  1.0597278 , -0.95991874, -1.41720255]])
    Y= np.array([[ 1.,  1., -1.,  1.,  1.,  1., -1., -1., -1., -1., -1.,  1.,  1.,
               1., -1., -1., -1.,  1.,  1., -1.]])
    return X, Y

# Used to define Xc, Yc in tests
def get_curve_data(random = False):
    X = np.array([[0.000000, 0.111111, 0.222222, 0.333333,
                   0.444444, 0.555556, 0.666667, 0.777778,
                   0.888889, 1.00000]])
    Y = np.array([[0.439369, 0.531638, 1.265424, 0.045298,
                   0.200480, -0.671454, -1.496848, -0.781617,
                   -0.529993, 0.01339]])
    return X, Y

# Test data
X1 = np.array([[1, 2, -1, 0],
              [3, -2, 1, 4]])
Y1 = np.array([[-1, 1, 1, 1]])
X2, Y2 = get_classify_data()
Xc, Yc = get_curve_data()


##------------------------LAB 6 WRITEUP --------------------------------

##1) Checkoff on kernels
##1.1) Radial Basis Kernel: weighs the labels of the training points based on
##                          their distance from the test point
X = np.array([[5, 7, 3, 8]])
Y = np.array([[-1, -1, 1, 1]])
alpha = np.array([[.5, 1, 1, .7]])
beta = 1

#RBF Prediction
z = np.array([[4]])
RBF = lambda x, z, beta: math.exp(-beta*(np.linalg.norm(x - z))**2)

def my_predict(X, Y, alpha, x, k, beta = None):# remove the +1 if it's a
                                            #kerne perceptron without offset
    kernel_x = (np.apply_along_axis(k, 0, X, x, beta) + 1)#1Xn array
    result = np.sum(np.multiply\
                    (np.multiply(Y, alpha), kernel_x))
    return result, np.sign(result)

z_pred = my_predict(X, Y, alpha, z, RBF, 1)
print (z_pred) #1 label for point z = 4


#1) 
#a) 0.38381638955625663
#b) 1

#2) with z's predicted label the data isn't linearly separable in the current
## feature space


#3) predictions [1, -1, -1, 1]

#5) infering what the separator was, given a graph of the sum of the
##(kernel*alpha*label)as a function of x. the sum  is equivalent to
##theta*x + theta_0. So, take two (sum, x) points from the graph and solve
## for theta, and theta_0 from the equation theta*x + theta_0 = sum(x)

## [-0.5, 2] #theta, theta_0

## 1.2) Discrete Feature Vectors

#the dataset is in R1. the values range between 1-10. they were discretized
#with width 1. phi(x) will have a large value (close to 1) on the dimension
#close to its original value in R1(i.e. its bucket and buckets closest to it)
#and a very small value (close to 0) on the dimensions that are further away

def dgphi(x, l = 0, m = 10, u = 10, beta = 1):
    w = (u - l) / m
    return np.array([np.exp(-beta * (x - w*i)**2) \
                     for i in range(m)]).reshape(-1,1)

#print (list(dgphi(2).reshape(1, 10)))

kernel_xz = lambda x, z: np.dot(dgphi(x).T, dgphi(z))

##print (kernel_xz(2,3), kernel_xz(2,5)) #[0.7492393 , 0.0137228]

##1.3) Radial Basis Function(RBF) Kernel

#The RBF kernel is the dot product of a continious version of the above
#discrete feature expansion  fuction where l and m range from -inf to inf and
#the width of 'discretization' is infinitely small

#i.e. we're using a feature vector that transforms x into an infinite dimen
#sional feature vector that looks like a gaussian centered on x


##1.4) String Theory
alphabet = 'abcdefghijklmnopqrstuvwxyz'
pats_phi = lambda y: np.array([[y.count(i) for i in alphabet]]).T
pats_kernel = lambda x,z : float(np.dot(x.T, z))

#1)
data = ["abalone", "xyzygy", "zigzag"]
training_pts = [pats_phi(word) for word in data]
y_tps = [10, 1, 3]
test_zig = pats_phi('ziggy')

kernel_ziggy = [pats_kernel(tp, test_zig) for tp in training_pts]
print (kernel_ziggy) #[0, 6, 7]


#2)
kernel_aba = [pats_kernel(tp, training_pts[0]) for tp in training_pts]
kernel_xyz = [pats_kernel(tp, training_pts[1]) for tp in training_pts]
kernel_zig = [pats_kernel(tp, training_pts[2]) for tp in training_pts]

print (kernel_aba, kernel_xyz, kernel_zig)
#matrix Axalpha = y
matrix_A = np.vstack((kernel_aba, kernel_xyz, kernel_zig))
print (matrix_A)

##1.5) Experimenting with kernel methods

#1.5.1/2) don't have access to the functions

#the point of the exercise was to see the effect of beta on the resulting
#classifier from an RBF kernel => beta = 1/variance. So, a small beta
#will give you a large variance and small bias and vice versa. If beta is
#small and point x_i is a support vector, then it will have a lot of say in
#the label of a test point even though it's far away from it

##2) Implementing Kernel Methods

#done above
















