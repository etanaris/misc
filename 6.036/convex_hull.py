##Convex Hull: Jarvis March
##
##1. find the left most point, x_o, in the dataset
##2. create a vector from x_0 to some point, x st x!=x_o, in the dataset (x_o ---> x)
##3. for every other point in the dataset, check where the point lies wrt the current vector
##        if it lies to the left:
##            pick that point as a convex hull point
##            update the current vector with that point (x_0 ---> x_left)
##        if it's colinear:
##            pick the point furthest away from x_o
##            update the current vector if necessary
##        if it lies to the right:
##            pass
##4. anchor the next vector at the convex hull point you ended up with from (3) and go back to to (1)
##5. done when (3) returns the initial left most point x_o as a convex hull point.
##
##Determining where a point (C) lies wrt some vector AB (A--->B): [in R2]
##    if the cross product AB X AC is positive:
##        the point C lies to the left
##    else:
##        it lies to the right

import numpy as np
import matplotlib.pyplot as plt
from itertools import product

#makes a numpy vector array from coordinate points
#start, end: tuples (x,y)
def vectorize(start, end):
    return np.array([start[0] - end[0], start[1] - end[1]])

#takes an array of points, and returns the list of points that make up the convex hull
#points: list of tuples (x,y) 
def convex_hull(points):
    left_most = min(points, key = lambda pt: pt[0])
    convex_hulls = [left_most]
    count = 0 
    while True:
        pt_0 = convex_hulls[-1] #tail of vector (must be a convex hull point)
        current_bound = None # tuple(vector, head of vector)  pt_0 ----> pt
        for pt in points:
            print ("----------------------- ROUND", count ,"-----------------------")
            print ('current covex hull point:', pt_0, 'current head point:', pt, \
                   'current vector:', current_bound)
            count +=1
            #vectorizing a pt with itself is just a null vector
            if pt == pt_0:
                print ('encountered the current convex hull point. passing \n')
                continue
            else:
                v = vectorize(pt_0, pt)
                
            #can't make a valid boundary with a convex_hull pt unless it's the left most one
            if pt in convex_hulls and pt != left_most:
                print ('encountered a non-left most convex_hull point. passing \n')
                continue
            
            #initalizing or updating the current boundary
            if current_bound == None:
                print ('current boundary vector not initalized. initializing the first boundary from', \
                       pt_0, 'to', pt, '\n')
                current_bound = (v, pt)
            else:
                if np.cross(current_bound[0], v) > 0:
                    print (pt, 'lies to the left of the current boundary vector. updating current boundary vector \n')
                    current_bound = (v, pt)
                elif np.cross(current_bound[0], v) == 0: #if the points are colinear, choose the furthest point
                    print ('found a colinear point:', pt, '\n')
                    current_bound = max ([current_bound, (v, pt)], key = lambda x: np.linalg.norm(x[0]))
                else:
                    print (pt, 'lies to the right of the current boundary vector. passing \n')


        convex_hulls.append(current_bound[1]) #for plotting reasons the left most point appears twice
        if current_bound[1] == left_most:
            print ('back to the left most point. done iterating \n')
            break
    return convex_hulls

def visualize(points):
    hulls = convex_hull(points)
    non_hulls = list(set(points).difference(set(hulls)))
    plt.show(plt.plot([i for i,j in hulls], [j for i,j in hulls], 'ro-',\
                      [i for i,j in non_hulls], [j for i,j in non_hulls], 'bo'))


##------------------TEST---------------- 

#test 1
square = [i for i in product([-1,1], repeat=2)]
square.append((0,0))
visualize(square)
#assert( set(square) == set(convex_hull(square)))

#test 2
line = [(1,1), (0,0), (-1,-1)]
visualize(line)
##assert( set([(1,1), (-1,-1)]) == set(convex_hull(line)))

#test 3
input1 = [(4.4, 14), (6.7, 15.25), (6.9, 12.8), (2.1, 11.1), (9.5, 14.9),\
          (13.2, 11.9), (10.3, 12.3), (6.8, 9.5), (3.3, 7.7), (0.6, 5.1), (5.3, 2.4),\
          (8.45, 4.7), (11.5, 9.6), (13.8, 7.3), (12.9, 3.1), (11, 1.1)]
output1 = [(13.8, 7.3), (13.2, 11.9), (9.5, 14.9), (6.7, 15.25), (4.4, 14),\
           (2.1,11.1), (0.6, 5.1), (5.3, 2.4), (11, 1.1), (12.9, 3.1)]
visualize(input1)
##assert( set(output1) == set(convex_hull(input1)))

#test 4
input2 = [(1, 0), (1, 1), (1, -1), (0.68957, 0.283647), (0.909487, 0.644276),\
          (0.0361877, 0.803816), (0.583004, 0.91555), (-0.748169, 0.210483),\
          (-0.553528, -0.967036), (0.316709, -0.153861), (-0.79267, 0.585945),\
          (-0.700164, -0.750994), (0.452273, -0.604434), (-0.79134, -0.249902), \
          (-0.594918, -0.397574), (-0.547371, -0.434041), (0.958132, -0.499614),\
          (0.039941, 0.0990732), (-0.891471, -0.464943), (0.513187, -0.457062), \
          (-0.930053, 0.60341), (0.656995, 0.854205)]

output2 = [(1, -1), (1, 1), (0.583004, 0.91555), (0.0361877, 0.803816),\
           (-0.930053, 0.60341), (-0.891471, -0.464943), (-0.700164, -0.750994),\
           (-0.553528, -0.967036)]
##assert ( set(output2) == set(convex_hull(input2)))
visualize(input2)






                    
                
    
    
