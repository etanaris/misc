import pdb
import numpy as np
import random
import matplotlib
import matplotlib.pyplot as plt

class NN:
    sizes = []
    act_funs = []
    act_deriv_funs = [] #a list of L−2 derivatives for the layer activation funcs
    loss_fun = None #a list of L-1 loss functions, one for each layer beyond the input
    loss_delta_fun = None
    class_fun = None
    dropout_pct = 0.

    def initialize(self):
        sigma = 1/np.sqrt(self.sizes[0]) #sizes[0] - units on the input layer
        mu = 0
        self.weights = [np.random.normal(mu, sigma, shape) for shape in\
                        zip(self.sizes[1:], self.sizes[:-1])]
        self.biases = [np.zeros((i,1)) for i in self.sizes[1:]] #biases for non-input layers
        return self

    def forward(self, x, drop=False):
        zs = []
        activations = [x]
        for i in range(len(self.sizes) - 1):
            z = np.dot(self.weights[i],activations[i]) + self.biases[i]
            a = self.act_funs[i](z)
            activations.append(a)
            zs.append(z)
        return (zs, activations)

    def backward(self, x, y):
        zs, actvns = self.forward(x)
        grad_bs = [self.loss_delta_fun(zs[-1], actvns[-1], y)]
        grad_ws = [np.dot(grad_bs[0], actvns[-2].T)]
        
        for i in range(len(self.sizes)- 3, -1, -1): #gradient for the hidden layers
            delta_l = np.dot(self.weights[i+1].T, grad_bs[0]) \
                      * self.act_deriv_funs[i](zs[:-1][i])
            grad_w = np.dot(delta_l, actvns[i].T)
            grad_bs.insert(0,delta_l)
            grad_ws.insert(0, grad_w)
        return (grad_ws, grad_bs)
    
    def evaluate(self, X, Y):
        count = 0
        loss = 0
        d, n = X.shape
        o, _ = Y.shape
        for i in range(n):
            zs, activations = self.forward(X[:,i:i+1])
            act_L = activations[-1]
            pj = self.class_fun(act_L)
            y = Y[:,i:i+1]
            yj = self.class_fun(y)
            if pj != yj:
                count += 1
            loss += self.loss_fun(act_L, y)
        return count/float(n), loss/float(n)
    
    def sgd_train(self, X, Y, n_iter, step_size, Xv=None, Yv=None):
        self.initialize()
        count = 0
        print ('MAX ITERS:', n_iter)
        min_error = 1
        while count < n_iter:
            idx = np.random.randint(X.shape[1]-1)
            grad_w, grad_b = self.backward(X[:,[idx]], Y[:,[idx]])
            self.weights = [w for w in map(lambda w, dw: w - (step_size*dw),\
                                           self.weights, grad_w)]
            self.biases = [b for b in map(lambda b, db: b - (step_size*db),\
                                          self.biases, grad_b)]
            count += 1
            error_rate = self.evaluate(X, Y)
            print ('Current Error Rate:', self.evaluate(X, Y))
            if error_rate[0] < min_error:
                min_error = error_rate[0]
            if error_rate[0] <= 0.005:
                break      
        print ('DONE TRAINING. Iterations taken:', count)
        print ('Final Error Rate:', self.evaluate(X, Y))
        print ('Min error_rate:', min_error)
        return self

def xor(N=200):
    # this is the XOR problem with softmax
    D = 2                               # dimension of data
    O = 2                               # dimension of output
    X = np.random.rand(D,N)             # we want [NxD] data
    X = (X > 0.5) * 1.0 
    Y = (X[0,:] == X[1,:])*1.0
    Y = np.vstack([Y, 1-Y])
    print(X); print(Y)
    # Add some noise to the data.
    X += np.random.randn(D,N)*0.2
    return X, Y

def relu(z):
    return np.maximum(z, np.zeros(z.shape))
def relu_deriv(z):
    return np.where(z > np.zeros(z.shape), np.ones(z.shape), np.zeros(z.shape))
def softmax(z):
    return  np.exp(z)/np.sum(np.exp(z))
def softmax_class(a):
    return np.argmax(a)
def nll(a, y):
    return -np.sum(y*np.log(a))
def nll_delta(z, a, y):
    return a - y

def hard():
    X= np.array([[-0.23390341,  1.18151883, -2.46493986,  1.55322202,  1.27621763,
              2.39710997, -1.3440304 , -0.46903436, -0.64673502, -1.44029872,
              -1.37537243,  1.05994811, -0.93311512,  1.02735575, -0.84138778,
              -2.22585412, -0.42591102,  1.03561105,  0.91125595, -2.26550369],
             [-0.92254932, -1.1030963 , -2.41956036, -1.15509002, -1.04805327,
              0.08717325,  0.8184725 , -0.75171045,  0.60664705,  0.80410947,
              -0.11600488,  1.03747218, -0.67210575,  0.99944446, -0.65559838,
              -0.40744784, -0.58367642,  1.0597278 , -0.95991874, -1.41720255]])
    Y= np.array([[ 1.,  1., 0.,  1.,  1.,  1., 0., 0., 0., 0., 0.,  1.,  1.,
                   1., 0., 0., 0.,  1.,  1., 0.]])
    Y = np.vstack([Y, 1-Y])
    print(X); print(Y)
    return X, Y

####################
# SUPPORT AND DISPLAY CODE
####################

# Takes a list of numbers and returns a row vector: 1 x n
def rv(value_list):
    return np.array([value_list])
def cv(value_list):
    return np.transpose(rv(value_list))

def tidy_plot(xmin, xmax, ymin, ymax, center = False, title = None,
                 xlabel = None, ylabel = None):
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

def plot_points(x, y, ax = None, clear = False, 
                  xmin = None, xmax = None, ymin = None, ymax = None,
                  style = 'or-', equal = False):
    padup = lambda v: v + 0.05 * abs(v)
    paddown = lambda v: v - 0.05 * abs(v)
    if ax is None:
        if xmin == None: xmin = paddown(np.min(x))
        if xmax == None: xmax = padup(np.max(x))
        if ymin == None: ymin = paddown(np.min(y))
        if ymax == None: ymax = padup(np.max(y))
        ax = tidy_plot(xmin, xmax, ymin, ymax)
        x_range = xmax - xmin; y_range = ymax - ymin
        if equal and .1 < x_range / y_range < 10:
            #ax.set_aspect('equal')
            plt.axis('equal')
            if x_range > y_range:
                ax.set_xlim((xmin, xmax))
            else:
                ax.set_ylim((ymin, ymax))
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    elif clear:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.clear()
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.plot(x, y, style, markeredgewidth=0.0, linewidth = 5.0)
    # Seems to occasionally mess up the limits
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    return ax

def add_ones(X):
    return np.vstack([X, np.ones(X.shape[1])])

def plot_data(data, labels, ax = None, 
                  xmin = None, xmax = None, ymin = None, ymax = None):
    # Handle 1D data
    if data.shape[0] == 1:
        data = add_ones(data)
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
    else:
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
    colors = np.choose(labels > 0, cv(['r', 'g']))[0]
    ax.scatter(data[0,:], data[1,:], c = colors,
                    marker = 'o', s=50, edgecolors = 'none')
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.grid(True, which='both')
    return ax

def plot_separator(ax, th, th_0):
    # If th is one dimensional, assume vertical separator
    if th.shape[0] == 1:
        th = cv([th[0, 0], 0.0])
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

def plot_objective_2d(J, ax = None, xmin = -5, xmax = 5,
                         ymin = -5, ymax = 5, 
                         cmin = None, cmax = None, res = 50):
    if ax is None:
        ax = tidy_plot(xmin, xmax, ymin, ymax)
    else:
        if xmin == None:
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
        else:
            ax.set_xlim((xmin, xmax))
            ax.set_ylim((ymin, ymax))

    ima = np.array([[J(cv([x1i, x2i])) \
                         for x1i in np.linspace(xmin, xmax, res)] \
                         for x2i in np.linspace(ymin, ymax, res)])
    im = ax.imshow(np.flipud(ima), interpolation = 'none',
                       extent = [xmin, xmax, ymin, ymax],
                       cmap = 'viridis')
    if cmin is not None or cmax is not None:
        if cmin is None: cmin = min(ima)
        if cmax is None: cmax = max(ima)
        im.set_clim(cmin, cmax)
    plt.colorbar(im)
    return ax



def classify(X, Y, hidden=[50, 50], it=10000, lr=0.005):
    D = X.shape[0]
    N = X.shape[1]
    O = Y.shape[0]
    # Create the network
    nn = NN()
    nn.sizes = [D] + list(hidden) + [O]
    nn.act_funs = [relu for l in list(hidden)] + [softmax]
    nn.act_deriv_funs = [relu_deriv for l in list(hidden)]
    nn.loss_fun = nll
    nn.loss_delta_fun = nll_delta
    nn.class_fun = softmax_class     # index of class
    # Modifies the weights and biases
    nn.sgd_train(X, Y, it, lr)

    # Draw it...
    def predict(x):
        return nn.class_fun(nn.forward(x)[1][-1])
    nax = plot_objective_2d(lambda x: predict(x),
                            xmin = -2.5, xmax = 2.5,  
                            ymin = -2.5, ymax = 2.5,
                            cmin = -1, cmax = 1)
    plot_data(X, Y, nax)
    plt.show()

    return nn

#------------TEST-------------
##iters_10, iters_50 = 0, 0
##n_trials = 50
##for trials in range(n_trials):
##    X,Y = hard()
##    iters_10 += classify(X, Y, hidden = [10, 10])[1]
##    iters_50 += classify(X, Y, hidden = [50, 50])[1]
##
##avg_iter_10 = iters_10/n_trials; avg_iter_50 = iters_50/n_trials
##print (avg_iter_10, avg_iter_50) #358.92 56.44

X,Y = xor()
##classify(X, Y, hidden = [10, 10])
##classify(X, Y, hidden = [50, 50])

##iters_10, iters_50 = 0, 0
##n_trials = 50
##for trials in range(n_trials):
##    X,Y = xor()
##    iters_10 += classify(X, Y, hidden = [10, 10])[1]
##    iters_50 += classify(X, Y, hidden = [50, 50])[1]
##
##avg_iter_10 = iters_10/n_trials; avg_iter_50 = iters_50/n_trials
##print (avg_iter_10, avg_iter_50) #lol your nerveeeeee
#------------------------------
    

'''
    # f_grad returns the loss functions gradient
    # x0 are the initial parameters (a starting point for the optimization)
    # data is a list of training data
    # args is a list or tuple of additional arguments passed to fgrad
    # stepsize is the global stepsize fir adagrad
    # fudge_factor is a small number to counter numerical instabiltiy
    # max_it is the number of iterations adagrad will run
    # minibatchsize if given is the number of training samples considered in each iteration
    # minibatch_ratio if minibatchsize is not set this ratio will be used to determine the batch size dependent on the length of the training data
    
    #d-dimensional vector representing diag(Gt) to store a running total of the squares of the gradients.
    gti=np.zeros(x0.shape[0])
    
    ld=len(data)
    if minibatchsize is None:
        minibatchsize = int(math.ceil(len(data)*minibatch_ratio))
    w=x0
    for t in range(max_it):
        s=sample(xrange(ld),minibatchsize)
        sd=[data[idx] for idx in s]
        grad=f_grad(w,sd,*args)
        gti+=grad**2
        adjusted_grad = grad / (fudge_factor + np.sqrt(gti))
        w = w - stepsize*adjusted_grad
    return w
'''

#-------------------------- LAB 7 Writeup ------------------------

#1) Neural Networks

#1.1) Crime and Punishment

#1) continious valued output (an average change in the stock market)

#number of units in the output layer: one
#activation funtion: linear
#loss function: hinge

#2)binary output: rains with probability p or not

#number of units in the output layer: two
#activation function: Relu
#loss function: NLL

#3)discrete non binary outputs: k buckets

#number of units in the output layer: k
#activation function: softmax
#loss function: NLL

#4)vector output: 1 if it addresses a topic 0 otherwise

#number of units in the output layer: number of topics
#activation function: Relu?
#loss function: hinge?

#1.2) Architecture

#m hidden units, no regularization : Model C
#2m hidden units, no regularizaton : Model A
#2m hidden units, dropout regularization : Model B



#2) Implementing neural networks

#done above

#3) Experiment
#error rate didn't get any better than 0.05

#dataset: hard ()
#best error_rate = 0.05
#max iters for 2 hidden layers of size 10: approx 360
#max iters for 2 hidden layers of size 50: approx 60 dayum

#dataset: xor()
#best error rate: 0.005
#max iters for 2 hidden layers of size 10: approx 5000
#max iters for 2 hidden layers of size 50: approx 1500


