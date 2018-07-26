import numpy as np
import hw4 


##1) Exploring gradient descent for regression and classification

##1.1) Least squares regression

##dimension of the first order polynomial basis K+1
## Don't have acces to t1. point is to drive home the idea that if you
## keep expanding to a higher dimension and try a linear regression model
## there, you will likely find a dimension where you have zero training error
## but it will generalize pretty poorly


#1.2) Regularizing the parameter vector
## Don't have access to the functions here as well.
## Figure out how the regularization param lambda affects how well
## a linear regression model performs. if the classification assumption holds
## then larger lambda implies smaller theta

##1.3) Gradient descent
##Don't have access to functions

##1.4) Pegasos

##stochiastic gradient descent (noisy updates save you from shallow local minimas

##mini-batch update: instead_d of updating over a single random point, update
#over k subsets of random points. 


##2)Linear regression

# Write a function that returns the gradient of lin_reg(x, th, th0)
# with respect to th
def d_lin_reg_th(x, th, th0):
    return x
    
# Write a function that returns the gradient of square_loss(x, y, th, th0) with
# respect to th.  It should be a one-line expression that uses lin_reg and
# d_lin_reg_th.
def d_square_loss_th(x, y, th, th0):
    return 2 * d_lin_reg_th(x, th, th0)*lin_reg(x, th, th0)

# Write a function that returns the gradient of mean_square_loss(x, y, th, th0) with
# respect to th.  It should be a one-line expression that uses d_square_loss_th.
def d_mean_square_loss_th(x, y, th, th0):
    return np.mean(d_square_loss_th(x, y, th, tho), axis = 1, keepdims = True)

##4)Stochastic gradient


def sgd(X, y, JdJ, w0, step_size_fn, max_iter):
    w = w0; fs = []; ws = []
    d, n = X.shape
    count = 0
    while (count < max_iter):
        idx = np.random.ranint(n)
        pt_x = X[:,[idx]]; label_x = y[idx]
        f, df = JdJ(pt_x, label_x) #df is a col vector
        fs.append(f); ws.append(w)
        w -= step_size_fn(count)*df
    return w, fs, ws #w[th'th0]


##5)Pegasos: stochastic gradient descent on the svm objective
def JdJ_lam(lam):
    # return a function JdJ that takes (X, y, w) and computes
    # svm loss and gradient wrt w (a weight vector [th; th0]) 
    # using the specified lam.
    def JDJ(X, y, w):
        f = lambda X, y, w, lam: hinge_loss(X, y, w[:-1], w[-1])\
            + lam * (np.linalg.norm(w))**2
        df = lambda X, Y, w, lam: np.vstack((d_hinge_loss_th(X, y, w[:-1], w[-1]),\

                                             d_hinge_loss_th0(X, y, w[:-1], w[-1])))\
                                             + 2*lam*np.linalg.norm(w)
    
        return f, df
    return JDJ
        
        

def Pegasos(data, labels, lam, max_iter):
    d, n = data.shape
    JdJ = JdJ_lam(lam)
    # Do not change these parameters
    w,_,_ = sgd(data, labels, JdJ, cv((d+1)*[0.]), lambda i: .1/(1+i),\
                max_iter)
    th, th0 = w[:-1],w[-1]
    return th, th0

        










