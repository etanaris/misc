# Implement perceptron, average perceptron, and pegasos
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import pdb
import itertools
import operator
import functools
from lab1 import *
from hw1 import perceptron as b_perceptron
from hw1 import misclassified


######################################################################
# Plotting

def tidy_plot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
    plt.ion()
    plt.figure(facecolor="white")
    ax = plt.subplot()
    if center:
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.spines['left'].set_smart_bounds(True)
        ax.spines['bottom'].set_smart_bounds(True)
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
    else:
        ax.spines["top"].set_visible(False)    
        ax.spines["right"].set_visible(False)    
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()
    eps = .05
    plt.xlim(xmin-eps, xmax+eps)
    plt.ylim(ymin-eps, ymax+eps)
    if title: ax.set_title(title)
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return ax

# LPK: replace this with something that will work even for vertical lines
#  and goes all the way to the boundaries
# Also draw a little normal vector in the positive direction
def plot_separator(ax, th, th_0):
    xmin, xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    pts = []
    eps = 1.0e-6
    # xmin boundary crossing is when xmin th[0] + y th[1] + th_0 = 0
    # that is, y = (-th_0 - xmin th[0]) / th[1]
    if abs(th[1,0]) > eps:
        pts += [np.array([x, (-th_0 - x * th[0,0]) / th[1,0]]) \
                                                        for x in (xmin, xmax)]
    if abs(th[0,0]) > 1.0e-6:
        pts += [np.array([(-th_0 - y * th[1,0]) / th[0,0], y]) \
                                                         for y in (ymin, ymax)]
    in_pts = []
    for p in pts:
        if (xmin-eps) <= p[0] <= (xmax+eps) and \
           (ymin-eps) <= p[1] <= (ymax+eps):
            duplicate = False
            for p1 in in_pts:
                if np.max(np.abs(p - p1)) < 1.0e-6:
                    duplicate = True
            if not duplicate:
                in_pts.append(p)
    if in_pts and len(in_pts) >= 2:
        # Plot separator
        vpts = np.vstack(in_pts)
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Plot normal
        vmid = 0.5*(in_pts[0] + in_pts[1])
        scale = np.sum(th*th)**0.5
        diff = in_pts[0] - in_pts[1]
        dist = max(xmax-xmin, ymax-ymin)
        vnrm = vmid + (dist/10)*(th.T[0]/scale)
        vpts = np.vstack([vmid, vnrm])
        ax.plot(vpts[:,0], vpts[:,1], 'k-', lw=2)
        # Try to keep limits from moving around
        ax.set_xlim((xmin, xmax))
        ax.set_ylim((ymin, ymax))
    else:
        print('Separator not in plot range')

def plot_data(data, labels, ax = None, clear = False,
                  xmin = None, xmax = None, ymin = None, ymax = None):
    if ax is None:
        if xmin == None: xmin = np.min(data[0, :]) - 0.5
        if xmax == None: xmax = np.max(data[0, :]) + 0.5
        if ymin == None: ymin = np.min(data[1, :]) - 0.5
        if ymax == None: ymax = np.max(data[1, :]) + 0.5
        ax = tidy_plot(xmin, xmax, ymin, ymax)

        x_range = xmax - xmin; y_range = ymax - ymin
        if .1 < x_range / y_range < 10:
            ax.set_aspect('equal')
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    colors = np.choose(labels > 0, cv(['r', 'g']))[0]
    ax.scatter(data[0,:], data[1,:], c = colors,
                    marker = 'o', s=50, edgecolors = 'none')
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    #ax.axhline(y=0, color='k')
    #ax.axvline(x=0, color='k')
    return ax

# Must either specify limits or existing ax
def plot_nonlin_sep(predictor, ax = None, xmin = None , xmax = None,
                        ymin = None, ymax = None, res = 30):
    if ax is None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    cmap = colors.ListedColormap(['black', 'white'])
    bounds=[-2,0,2]
    norm = colors.BoundaryNorm(bounds, cmap.N)            
            
    ima = np.array([[predictor(x1i, x2i) \
                         for x1i in np.linspace(xmin, xmax, res)] \
                         for x2i in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none',
                       extent = [xmin, xmax, ymin, ymax],
                       cmap = cmap, norm = norm)

######################################################################
#   Utilities

# Takes a list of numbers and returns a column vector:  n x 1
def cv(value_list):
    return np.transpose(rv(value_list))

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])

# x is dimension d by 1
# th is dimension d by 1
# th0 is a scalar
# return a 1 by 1 matrix
def y(x, th, th0):
   return np.dot(np.transpose(th), x) + th0

# x is dimension d by 1
# th is dimension d by 1
# th0 is dimension 1 by 1
# return 1 by 1 matrix of +1, 0, -1
def positive(x, th, th0):
   return np.sign(y(x, th, th0))

# data is dimension d by n
# labels is dimension 1 by n
# ths is dimension d by 1
# th0s is dimension 1 by 1
# return 1 by 1 matrix of integer indicating number of data points correct for
# each separator.
def score(data, labels, th, th0):
   return np.sum(positive(data, th, th0) == labels)

######################################################################
#   Data Sets

# Return d = 2 by n = 4 data matrix and 1 x n = 4 label matrix
def super_simple_separable_through_origin():
    X = np.array([[2, 3, 9, 12],
                  [5, 1, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def super_simple_separable():
    X = np.array([[2, 3, 9, 12],
                  [5, 2, 6, 5]])
    y = np.array([[1, -1, 1, -1]])
    return X, y

def xor():
    X = np.array([[1, 2, 1, 2],
                  [1, 2, 2, 1]])
    y = np.array([[1, 1, -1, -1]])
    return X, y

def xor_more():
    X = np.array([[1, 2, 1, 2, 2, 4, 1, 3],
                  [1, 2, 2, 1, 3, 1, 3, 3]])
    y = np.array([[1, 1, -1, -1, 1, 1, -1, -1]])
    return X, y

######################################################################
#   Tests for part 2:  features

# Make it take miscellaneous args and pass into learner
def test_linear_classifier_with_features(dataFun, learner, feature_fun,
                             draw = True, refresh = True, pause = False):
    raw_data, labels = dataFun()
    data = feature_fun(raw_data) if feature_fun else raw_data
    if draw:
        ax = plot_data(raw_data, labels)
        def hook(params):
            (th, th0) = params
            plot_nonlin_sep(
                lambda x1,x2: int(positive(feature_fun(cv([x1, x2])), th, th0)),
                ax = ax)
            plot_data(raw_data, labels, ax)
            print('th', th.T, 'th0', th0)
            if pause: input('go?')
    else:
        hook = None
    th, th0 = learner(data, labels, hook = hook)
    if hook: hook((th, th0))
    print("Final score", int(score(data, labels, th, th0)))
    print("Params", np.transpose(th), th0)
    return th, th0

def mul(seq):
    return functools.reduce(operator.mul, seq, 1)

def make_polynomial_feature_fun(order):
    # raw_features is d by n
    # return is D by n where D = sum_{i = 0}^order  multichoose(d, i)
    def f(raw_features):
        d, n = raw_features.shape
        result = []   # list of column vectors
        for j in range(n):
            features = []
            for o in range(order+1):
                indexTuples = \
                          itertools.combinations_with_replacement(range(d), o)
                for it in indexTuples:
                    features.append(mul(raw_features[i, j] for i in it))
            result.append(cv(features))
        return np.hstack(result)
    return f

def test_with_features(dataFun, order = 2, draw=True):
    return test_linear_classifier_with_features(
        dataFun,                        # data
        perceptron,                     # learner
        make_polynomial_feature_fun(order), # feature maker
        draw=draw)
##--------------------------HOMEWORK 2-------------------------
#1)XOR

#1.1. Part 1

def phi(x_i):
    return np.vstack((x_i, x_i**2))

data = np.array([[1, 1, 2, 2],[1, 2, 1, 2]])
labels = np.array([[-1, 1, 1, -1]])
transformed = np.hstack(tuple(phi(data[:,[i]]) for i in range(data.shape[1])))
##print (transformed)

#1.2. Part 2

##print (perceptron(transformed, labels))

#2)SCALING

data_2 = np.array([[200, 800, 200, 800], [.2, .2, .8, .8], [1]*4])
labels_2 = np.array([[-1, -1, 1, 1]])
theta = (np.array([[0,1, -0.5]])).T
theta_norm = np.linalg.norm(theta)
x_i_norms = [np.linalg.norm(data_2[:,[i]]) for i in range(data_2.shape[1])]
R = max(x_i_norms) #800.0010249993434
gammas = [(labels_2[0][i] *((np.dot(theta.T, data_2[:,[i]]))))/theta_norm\
          for i in range(data_2.shape[1])] #every point seems equidistant
#2.1.
gamma = min(gammas) # 0.26832816
#print (gammas, gamma)

#2.2. from hw1.py

perceptron_bound = (R/gamma)**2 #8888911.67
##print (R, gamma, perceptron_bound)

#2.3. from hw1.pyperceptron

##data_2 = [((np.array([[200,.2]])).T ,-1),((np.array([[800,.2]])).T, -1), \
##         ((np.array([[200,.8]])).T, 1), ((np.array([[800,.8]])).T, 1)]
##
##avg_convergence_2 = []
##for i in range(10):
##    avg_convergence_2.append(b_perceptron(data_2, 0, offset = True))
##
##print (avg_convergence_2)
##irl_perceptron_rounds = sum([i for a,b,i in avg_convergence_2])/10
##print (irl_perceptron_rounds) #833376

#2.4./2.5.
data_3 = np.copy(data_2)
data_3[0:2]*=0.001
theta_3 = (np.array([[0,1, -0.0005]])).T
theta_3_norm = np.linalg.norm(theta_3)
gammas_3 = [(labels_2[0][i] *((np.dot(theta_3.T, data_3[:,[i]]))))/theta_3_norm\
          for i in range(data_3.shape[1])] #every point is equidistant
gamma_3 = min(gammas_3) #0.0003
#print (gammas_3, gamma_3)
#gamma is even smaller now so it will take perceptron much longer to converge

#2.6.
data_4= np.copy(data_2)
data_4[0]*=0.001
gammas_4 = [(labels_2[0][i] *((np.dot(theta.T, data_4[:,[i]]))))/theta_norm\
          for i in range(data_4.shape[1])]
gamma_4 = min(gammas_4) # 0.26832816
x_4_norms = [np.linalg.norm(data_4[:,[i]]) for i in range(data_4.shape[1])]
R_4 = max(x_4_norms) # 1.50996688705415
##print (gammas_4, gamma_4)

#2.7.
##print (R_4, gamma_4, (R_4/gamma_4)**2) #perceptron bound = 31.66666667

#2.8. from hw1.py

##data_4 = [((np.array([[0.2,.2]])).T ,-1),((np.array([[0.8,.2]])).T, -1), \
##         ((np.array([[0.2,.8]])).T, 1), ((np.array([[0.8,.8]])).T, 1)]
##
##avg_convergence_4 = []
##for i in range(10):
##    avg_convergence_4.append(b_perceptron(data_4, 0, offset = True))
##irl_perceptron_rounds_scaled = sum([i for a,b,i in avg_convergence_4])/10
##print (irl_perceptron_rounds_scaled) #16

#now since the first feature has been scaled down, R is much smaller and it will
#take much less time for perceptron to converge

#3)ENCODING DISCRETE VALUES

#1. did this in lab1.py under perceptron
# theta, theta_o = (array([[-2.]]), 7)

#2./3. 
#samsung = 1 , nokia = 6
pts = [1,6]
#[+1, -1], yes this makes sense


#4./5. 
data_unencoded = np.array([[2, 3,  4,  5]])
label_ = np.array([[1, 1, -1, -1]])
##print (perceptron(data_unencoded, label_))

def one_hot(x, k):
    col_vector = np.zeros((k,1)); col_vector[x-1] = 1
    return col_vector
data_encoded = np.hstack(tuple(one_hot(data_unencoded[0][i], 6) for i in range(data_unencoded.shape[1])))
##print (data_encoded)
theta, theta_o = perceptron(data_encoded, label_)
##print (theta, theta_o)

###predict for samsung = 1, and nokia = 6
points= [1,6]
predict =[]
for pt in points:
    predict.append(np.sign(np.dot(theta.T, one_hot(pt, 6))+ theta_o))
##print (predict)
# prediction = [0,0] , i think i did something wrong but i don't see it

encoded_pts = [one_hot(i, 6) for i in pts]
def signed_dist(x, th, th_0):
    return (np.dot(th.T, x) + th_0)/np.linalg.norm(th)
##for en_pt in encoded_pts:
##    print (signed_dist(en_pt, theta, theta_o))  #they apparently lie on the separator.

#6./7./8./ did this in lab1.py as well
#it's not linearly separable in the original encoding
#it is linearly separable after the encoding tho
#theta, theta_0 = [ 1., 1. , -2., -2., 1. , 1.] , 0

data =np.array([[1, 2, 3, 4, 5, 6]]) #not linearly separable
labels = np.array([[1, 1, -1, -1, 1, 1]])
th, tho = perceptron(data, labels)
##print (data.shape[1], score(data, labels, th, tho)) #training_error = 0.5, not linearly separable

data_en = np.hstack\
               (tuple(one_hot(data[0][i], 6) for i in range(data.shape[1])))
##print (data_en)
th_en, tho_en = perceptron(data_en, labels)
##print (th_en.T, tho_en)
##print (data_en.shape[1], score(data_en, labels, th_en, tho_en)) #training_error = 0, linearly separable



#3)FEATURE VECTORS

#3.1)

#a) A = is a 2X6 matrix
#A = np.array([[1/6]*6, [1/3]*3+[-1/3]*3])


#b) theta_x = np.dot(A.T, theta_z) #6x2.2x1 = 6x1


#c) a linear classifier existing in a higher dimension does not guarantee
#the data points being linearly separable in a lower dimension. However,
#if you have a linear classifer in a lower dimension then a linear classifier
#definitely exists in a higher dimension as well. So, no.

#d) it will converge faster in z-space because the upper bound R is smaller

#z-space is R(1), orginal x-space is in R(2) (done with hw1.py perceptron)

#perceptron in z space aint doing any better...
A = np.array([[0.01,0.1]])
data_x = [(np.array([[100, -100]]).T, -1), (np.array([[-100, 100]]).T, 1)]
data_z = [(np.dot(A, data_x[0][0]), -1),(np.dot(A, data_x[1][0]), 1)]

##print (data_x, data_z)
##print (b_perceptron(data_x, 0))
##print (b_perceptron(data_z, 0))

#e) training in a lower feature space will generalize well since it won't
#overfit the data but it's possible that it will have a higher training error
#than training in a higher dimension where you can have zero training error
#but more likely to overfit the data ==> higher testing/generalization error


#4)POLYNOMIAL FEATURES


#1. orignal spaces in R(2) and R(3)

order = [0, 1, 10, 20, 30, 40, 50]
x_2 = np.array([[1, 1]]).T ; x_3 = np.array([[1, 1, 1]]).T
features_2 = [] ; features_3 = []
for o in order:
    features_2.append(make_polynomial_feature_fun(o)(x_2).shape[0])
    features_3.append(make_polynomial_feature_fun(o)(x_3).shape[0])
    
##print (features_2, features_3)
# [3, 66, 231, 496, 861, 1326], [4, 286, 1771, 5456, 12341, 23426]


#2.

data_funs = [super_simple_separable_through_origin, super_simple_separable,\
             xor, xor_more]

#test_with_features(dataFun, order = 2, draw=True)
def min_order(data_funs):
    min_orders = []
    for data_fun in data_funs:
        order = 0
        data_init, labels = data_fun()
        while True:
            data = make_polynomial_feature_fun(order)(data_init)
            th, th0 = test_with_features(data_fun, order, False)
            if score(data, labels, th, th0)== data.shape[1]:
                break
            order+=1
        min_orders.append(order)
    return min_orders

##print (min_order(data_funs)) #[1, 1, 2, 3]
    











