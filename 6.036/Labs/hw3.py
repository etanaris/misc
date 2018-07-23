import numpy as np

#week 3 exercise

def signed_dist(x, th, th_0):
    return (np.dot(th.T, x) + th_0)/np.linalg.norm(th)

def find_margin(data,th, th0, labels):
    return labels * signed_dist(data, th, th0)

    
##data = np.array([[3,1,4],[2,1,2]])
##labels = np.array([[1, -1, -1]])
##th = np.array([[1,1]]).T
##th0= -4

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



##1)FAR FROM THE MADDING CROWD

##data = np.array([[1, 2, 1, 2, 10, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7],
##                 [1, 1, 2, 2,  2,  2,  2,  2,  2,  2,  2,  2]])
##labels = np.array([[-1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
##blue_th = np.array([[0, 1]]).T
##blue_th0 = -1.5
##red_th = np.array([[1, 0]]).T
##red_th0 = -2.5
##
##blue_margins = find_margin(data, blue_th, blue_th0, labels)[0]
##red_margins = find_margin(data, red_th, red_th0, labels)[0]
##
##red_S_sum, blue_S_sum = np.sum(red_margins), np.sum(blue_margins)
##red_S_min, blue_S_min = np.min(red_margins), np.min(blue_margins)
##red_S_max, blue_S_max = np.max(red_margins), np.max(blue_margins)
##
##print ('RED: ' , red_margins, red_S_sum, red_S_min, red_S_max)
##print ('BLUE:' , blue_margins, blue_S_sum, blue_S_min, blue_S_max)
                               
##2)WHAT  A LOSS

##3)SIMPLY INSEPARABLE

#2.

data = np.array([[1.1, 1, 4],[3.1, 1, 2]])
labels = np.array([[1, -1, -1]])
th = np.array([[1, 1]]).T
th0 = -4
gamma_ref = np.sqrt(2)/2
margins = find_margin(data, th, th0, labels)[0]
hinge_loss = [max(0, 1- (margin/gamma_ref)) for margin in margins]
##print ('HINGE LOSS:' , hinge_loss) #[0.7999999999999998, 0, 3.0]



##4)IT HINGES ON THE LOSS

##gamma_ref = 1/norm(theta). Therefore, minimizing theta means maximizing
##gamma_ref, the minimum allowable margin of any point in the data set from
##the separator. This is achieved by including the regularization function
##(with positve lambda) as part of the SVM objective.

##5)LIMITS OF LAMBDA

data = np.array([[1,1,1],[1,4,3], [1,1,2]])
labels = np.array([[-1, -1, 1]])
##objective function = gradJ wrt theta=> 0+ 2*lambda*theta if right
## gradJ wrt theta => y_i*x_i + 2*lambda*theta

def SVM_with_regularization(data, labels,lamda, iters = 100 ):
    d, n = data.shape
    theta = (np.array([[0.0]*d])).T
    labels = labels.T
    for i in range(iters):
        for i in range(n):
            theta_norm = np.linalg.norm(theta)
            if (np.dot(theta.T, data[:,[i]]))*labels[i] <=1: #misclassified               
                theta += (data[:,[i]]*labels[i]) + 2*lamda*theta_norm
            else: #correctily classified, just add regulariation
                theta +=2*lamda*theta_norm
    return theta

##lambdas = [0 , 0.001, 0.02]
##for l in lambdas:
##    print (SVM_with_regularization(data, labels, l)) #not functional

#l2-regularization serves to minimize and equalize all the dimensions of theta. theta and lambda are
#are inversly related. so, a large lambda implies a small theta. A smaller theta implies a larger
#gamma ref (minimum max_margin). so, the better the max margin of the separator is, the larger the regularization
#parameter


##6.LINEAR SUPPORT VECTOR MACHINES




           







