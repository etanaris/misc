import numpy as np

def length(col):
    return np.sqrt(np.sum(col*col))

def normalize(col_v):
    return (1/length(col_v))*col_v

#point from a hyperplane distance and direction
def signed_dist(x, th, th_0):
    return (np.dot(th.T, x) + th_0)/length(th)

#a single d*1 data point
def positive_1(x, th, th_0):
    return np.sign(signed_dist(x, th, th_0))

#multiple(n) d*n data points (postiive or negative side of hyperplane
def positive_n(data, th, th_0):
    return np.apply_along_axis(positive_1, 0, data, th, th_0 )

#how many correctly classified points by the hyperplane
def score(data, labels, th, th_0):
    A = lables == np.apply_along_axis(positive, 0, data, th, th_0)
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



#week 1 exercises
#1)
data = np.transpose(np.array([[1, -1,2,-3],[1,2,3,4],[-1,-1,-1,-1],[1,1,1,1]]))
theta = np.transpose(np.array([[1, -1, 2, -3]]))
th_0 = 0
print (data, positive_n(data, theta, th_0))

#2)

