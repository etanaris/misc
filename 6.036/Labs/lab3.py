import numpy as np
import matplotlib.pyplot as plt
from lab1 import *
##import sys

##1)GRADIENT DESCENT

##1.1) Explore gradient descent 
#function : (2x +3)**2
def t1(step_size= 0.1, init_value= 0, eps = 0.00001, iters= 1000):
    x = init_value
    count = 0
    oscillated = []
    while (count < iters):
        count +=1
        x_prev = x
        x-= (step_size*(8*x+12))
        oscillated.append(x < -1.5) 
        if (abs(x - x_prev) <= eps):
            break
    return x, any(oscillated)


step_sizes = [i*0.01 for i in range(24)]
optimal_xs = [t1(step_size = s)[0] for s in step_sizes]
optimal_ys = [(2*x + 3)**2 for x in optimal_xs]
xs = [i for i in range(-20, 20)]
ys = [(2*x + 3)**2 for x in xs]
##plt.show(plt.plot(xs, ys, 'bo-', optimal_xs, optimal_ys, 'ro'))


non_osil = [val for val in filter(lambda x: x[0][1] == False,\
                  [(t1(step_size = s), s) for s in step_sizes])]
osil= [val for val in filter (lambda x: x[0][1] == True, \
               [(t1(step_size = s), s) for s in step_sizes])]
##print ('NO_OSCILLATE:', non_osil) #2. step_size = 0.12
##print ('OSCILLATE:', osil) #3. step_size = 0.13
##print ('DIVERGE:', t1(step_size = 0.26)) #4. step_size = 0.26
##print ('NO DIVERGE BUT SUBOPTIMAL:', t1(step_size = 0.2497)) #5. step_size = 0.2497
##print (t1(init_value = 100000000000000)) #6. extremely large +/- values didn't work..

##1.2- 1.4)don't have access to the functions

##2)IMPLEMENTING GRADIENT DESCENT

##2.1) Gradient descent
##f(x) is a scalar function, x is in Rd

##fdf: a function whose input is an x, a column vector,and returns a tuple
##(f, df), where f is a scalar and df is a column vector representing the gradient of f at x.
##x0: an initial value of xxx, x0, which is a column vector.
##step_size_fn: a function that is given the iteration index (an integer)
#and returns a step size.
##max_iter: the number of iterations to perform

def gd(fdf, x0, step_size_fn, max_iter):
    x = x0; fs = []; xs = []
    iters = 0
    while (iters < max_iter):
        fx, grad_fx = fdf(x)
        xs.append(x); fs.append(fx)
        x -= step_size_fn(iters)* grad_fx
        iters+=1
    return (x, fs, xs)

##2.2) Numerical gradient

#f is an objective function that takes a scalar x?
#if f takes a col vector simply feed x-deltas and x+deltas to f instead of
#np.apply_along_axis
def num_grad(f, delta=0.001):
    def grad_fx(x):
        d = x.shape[0]
        deltas = np.ones((d,1))*delta
        fx_minusdelta = f(x-deltas) #np.apply_along_axis(f, 1, x-deltas)
        fx_plusdelta = f(x+deltaas) #np.apply_along_axis(f, 1, x+deltas)
        return np.apply_along_axis\
               (lambda i: i/(2*delta), 1, fx_plusdelta - fx_minusdelta)
    return grad_fx

##2.3) Using the numerical gradient
#if f takes x component by component then define a function inside that
#takes x and applies f along x(the col vector) and use that function instead
def minimize(f, x0, step_size_fn, max_iter):
    grad_x = num_grad(f)
    def my_fdf(x):
        return f(x),grad_x(x)
    return gd(my_fdf, x0, step_size_fn, max_iter)







        










        
