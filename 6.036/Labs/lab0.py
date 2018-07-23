import numpy as np

def length(col):
    return np.sqrt(np.sum(col*col))

def normalize(col_v):
    return (1/length(col_v))*col_v

def signed_dist(x, th, th_0):
    return (np.dot(th.T, x) + th_0)/length(th)
    

#a single d*1 data point
def positive_1(x, th, th_0):
    return np.sign(signed_dist(x, th, th_0))

#multiple(n) d*n data points
def positive_n(data, th, th_0):
    return np.apply_along_axis(positive_1, 0, data, th, th_0 )

def score(data, labels, th, th_0):
    A = lables == np.apply_along_axis(positive_1, 0, data, th, th_0)
    return np.sum(A)

# still no loops????
def scores(data, labels, ths, th_0s):
    output = []
    for c in range(ths.shape[1]):
        A = lables == np.apply_along_axis(positive, 0, data, ths[:,c], th_0s[:,c])
        output.append([np.sum(A), ths[:,c], th_0s])
    return output
    
def best_separator(data, labels, ths, th0s):
    return tuple(max(scores(data, labels, ths, th0s), key = lambda x: x[0])[:3])


#week 1 exercise
#1.
##data = np.transpose(np.array([[1,-1,2,-3],[1,2,3,4],[-1,-1,-1,-1], [1,1,1,1]]))
##theta = np.transpose(np.array([[1,-1,2,-3]]))
##th_0 = 0
##print (positive_n(data, theta, th_0))

#week 3 exercise

def find_margin(data,th, th0, labels):
    return labels * signed_dist(data, th, th0)

    
##data = np.array([[3,1,4],[2,1,2]])
##labels = np.array([[1, -1, -1]])
##th = np.array([[1,1]]).T
##th0= -4
##
##print (find_margin(data, th, th0, labels)) #[0.70710678, 1.41421356, -1.41421356]
##
##data_2= np.array([[1, 1, 3, 3],[3, 1, 4, 2]])
##labels_2 = np.array([[-1, -1, 1, 1]])
##th_2 = np.array([[0, 1]]).T
##th0_2 = -3
##
##print (find_margin(data_2, th_2, th0_2, labels_2)) #[-0.  2.  1. -1.]
##
##th_max= np.array([[1, 0]]).T
##th0_max = -2
##
##print (find_margin(data_2, th_max, th0_max, labels_2)) #[1. 1. 1. 1.]

