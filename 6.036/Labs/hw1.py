import numpy as np
import matplotlib.pylab as plt
from itertools import product


#do you have a reason for this tho gurl
##plt.show(plt.plot([1,-10], [-1,-1], 'bo', [0], [1], 'ro'))


#1. PERCEPTRON MISTAKES
#returns true if a data pt is misclassified by the given theta, and theta_o
def misclassified(data_pt, label, theta, theta_o):
##    print (np.dot(theta.T, data_pt) + theta_o)
    return np.sign(np.dot(theta.T, data_pt) + theta_o)!= label


#data is a list of tuples of (np.col_vector(x,y), label)
def perceptron(data, start, offset = False):
    #initalizing theta, theta_o, the miss and round counters
    dim = data[0][0].shape[0] #number of rows in the col vector
    null = [0.0] * dim
    theta = (np.array([null])).T
##    theta = (np.array([[1000.0, -1000.0]])).T #4)initalization
    theta_o = 0
    count_miss = dict()
    rounds = 0
    while True:
        for vector,label in data[start:]+data[:start]:
            rounds +=1
            if misclassified(vector, label, theta, theta_o):
                #updating rounds of perceptron and misses for each point
                misses = count_miss.get(str(vector), 0)
                count_miss[str(vector)] = misses + 1
                
                #updating theta and theta_o
                theta += (vector*label)
                if offset:
                    theta_o += label
        if all(map(lambda entry: not misclassified(entry[0], entry[1], theta, theta_o), data)):
            break  
    return theta, theta_o,rounds ,count_miss

#2)SCALING: HOMEWORK 2: TESTING REAL PERCEPTRON BOUND

##data_2 = [((np.array([[200,.2]])).T ,-1),((np.array([[800,.2]])).T, -1), \
##         ((np.array([[200,.8]])).T, 1), ((np.array([[800,.8]])).T, 1)]
##
##data_4 = [((np.array([[0.2,.2]])).T ,-1),((np.array([[0.8,.2]])).T, -1), \
##         ((np.array([[0.2,.8]])).T, 1), ((np.array([[0.8,.8]])).T, 1)]
##
##avg_convergence_2 = []
##for i in range(10):
##    avg_convergence_2.append(perceptron(data_2, 0, offset = True))    
##print (avg_convergence_2)
##print (sum([i for a,b,i in avg_convergence_2])/10)
##
##avg_convergence_4 = []
##for i in range(10):
##    avg_convergence_4.append(perceptron(data_4, 0, offset = True))    
##print (avg_convergence_4)
##print (sum([i for a,b,i in avg_convergence_4])/10)


###1.PERCEPTRON MISTAKES
##data = [((np.array([[1,-1]])).T ,1),((np.array([[0,1]])).T, -1), \
##         ((np.array([[-10,-1]])).T, 1)]
##print (perceptron(data, 0))
###start 0
##print (perceptron(data, 0))
#start 1
##print (perceptron(data, 1))
#sanity check
theta = (np.array([[0, -1]])).T
##print ([i for i in map(lambda data_pt: misclassified(data_pt[0], data_pt[1], theta,0), data)])

#2.DUAL VIEW

data_1 = [((np.array([[-3,2]])).T ,1),((np.array([[-1,1]])).T, -1), \
         ((np.array([[-1,-1]])).T, -1), ((np.array([[2,2]])).T, -1), \
         ((np.array([[1,-1]])).T, -1)]

## All return the same theta and number of rounds
##print (perceptron(data_1, 0, True)) #this is the one. 
##print (perceptron(data_1, 1, True)) #this returned a different theta_o from the rest
##print (perceptron(data_1, 2, True))
##print (perceptron(data_1, 3, True))


#3.DECISION BOUNDARIES
#3.1.And
x = [i for i in product((0,1), repeat= 3)]
f_x = [(x_1 & x_2 & x_3) for x_1,x_2,x_3 in x]
label_x = [ -1 if i==0 else 1 for i in f_x]
numpy_x = [(np.array([list(i)])).T for i in x]
data_2 = [i for i in zip(numpy_x, label_x)]

##print(perceptron(data_2, 0, True))
##print(perceptron(data_2, 0)) #stalls


#3.2.Families

points = [(-1,1), (1,-1), (1,1), (2,2)]
labels = [1, 1, -1, -1]

###1)
dist_from_origin = [np.sqrt(x**2 + y**2) for x,y in points]
##print (dist_from_origin)
### [1.4142135623730951, 1.4142135623730951,1.4142135623730951, 2.8284271247461903]
### since there are points equidistant from the origin with different labels
### there can't be a circular classifier for those points centered at the orgigin

###2)
dist_from_2_2 = [np.sqrt((x-2)**2 + (y-2)**2) for x,y in points]
##print (dist_from_2_2)
### [3.1622776601683795, 3.1622776601683795, 1.4142135623730951, 0.0]
### This can be classified by a circle centered at (2,2) with r < 3.16 

###3)
numpy_pts = [(np.array([list(i)])).T for i in points]
data_3 = [i for i in zip(numpy_pts, labels)]
##print (data_3)
##print (perceptron(data_3, 0)) #stalls
###Can't be classified by a line through the origin

###4)
##print (perceptron(data_3, 0, True))
###can be classified by theta [-1, -1] and theta_o = 1

#4.INITIALIZATION
##print (perceptron(data, 0))
###sanity check
##crazy_theta = (np.array([[769., -1154.]])).T 
##print ([i for i in map(lambda data_pt: misclassified(data_pt[0], data_pt[1], crazy_theta, 0), data)])
###since theta starts far off from where any of the data points are, it'll take much longer

#5.MISTAKES AND GENERALIZATION
#5.2 Mistake Bound
##gammas = [.00001, .0001, .001, .01, .1, .2, .4]
##R = 1
##print ([(R/gamma)**2 for gamma in gammas])
###[9999999999.999996, 100000000.0, 1000000.0, 10000.0, 100.0, 25.0, 6.25]
##Eyeballed the rest
#R- upper bound on the data point's magneitude
#gamma - the smallest distance of a data point from the decision boundary

#6.SEPARATION

#takes data points that are not linearly seperable by a decision boundary through the origin
#but seperable by an offset and returns a new data set such that running perceptron without
#offset on this new data set should enable us to find a separator for the original data set
#data is a dXn array and labels is a 1Xn array
#row of 1s on the original data, that accounts for the theta_o.
def new_data(data, labels):
    n = data.shape[1]
    return np.vstack((np.array([[1]*n]), data)), labels

##n = 4, d = 4
##labels = (np.array([[1]*4])).T
##print (new_data(np.eye(4), labels)[0].shape)

