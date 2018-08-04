import numpy as np

#1) Loss functions and output activations: classification
#1.1) Hinge loss, linear activation


#x: col vector
#a: a number, an activation
#y: a number, a label
#d_dwi = d_da*da_dwi : da_dwi = x_i
def linear_hinge_grad(x, y, a):
    d_da = -y if 1-y*a <=1 else 0
    return x*d_da

##1.2) log loss, sigmoidal activation

#d_sigmoid(x) = sigm(x)(1 - sigm(x))

#activation from a neural unit is interprated as a probability of its label being
#positive. objective of learning is to maximize P(y,a). The labels now switch
#to y E (1,0). this way the probability of the label and the activation
#can be represented as a multiple of a**y*(1-1)**(1-y) over all the units in
#the penultimate layer?


#2) Multiclass (non-binary) Classification

#Softmax(SM) activation function

def SM(inpt):
    inter = np.exp(inpt)
    total = np.sum(inter)
    return inter/total
z_L = np.array([[-1, 0, 1]])
print (SM(z_L))


def NLL(a,y):
    return -sum([i*np.log10(j) for i,j in zip(y,a)])

a=[.3,.5,.2]
y=[0,0,1]

print (NLL(a,y))

w_L = np.array([[1, -1, -2],[-1, 2, 1]])
x = np.array([[1,1]]).T
y = np.array([[0, 1, 0]]).T

def grad_NLL(y, x, w_L):
    z = np.dot(x.T, w_L)
    a_L = SM(z)
    print (a_L)
    return np.dot(x, a_L - y.T)

print (grad_NLL(y, x, w_L))

eta = 0.5
new_wL = w_L - (eta*grad_NLL(y,x,w_L))
print (new_wL)
new_z = np.dot(x.T, new_wL)
new_aL = SM(new_z)
print (new_aL)


#3) LINEAR + LINEAR  = LINEAR (Notes)


#4) Neural Networks

#ReLU activation fun on all non-input layers. softmax fun on the last hidden
#layer for the final output

#4.1) Output

ReLU = np.vectorize(lambda x: max(x,0))
W = np.array([[1,0, -1], [0,1,-1], [-1, 0, -1], [0, -1, -1]])
V = np.array([[1,1,1,1, 0], [-1,-1,-1,-1,2]])
x = np.array([[3, 14, 1]])
z = np.dot(x, W.T)
f_z = ReLU(z) # first layer
fz = np.hstack((f_z, np.array([[1]]))) #added 1 for the offset
print ('Z', z, 'F_Z:', f_z)
u = np.dot(fz, V.T)
f_u = ReLU(z) #second layer
sm_out = SM(f_u) #final output with softmax
##print ('U', u, 'Fu', f_u)
print ('SM', sm_out )

#4.2) Unit decision boundaries

X = np.array([[0.5, 0.5 , 1],[0, 2, 1],[-3, 0.5, 1]])
z = np.dot(X, W.T)
fz_s = ReLU(z)
print ('Z', z, 'FZ_s', fz_s)

#4.3) Network outputs

#assume you still have the given V values for the last layer
#u_1 = fz_1 + fz_2 + fz_3 + fz_4
#u_2 = -(fz_1 + fz_2 + fz_3 + fz_4) + 2

#case 1)  fz_1 + fz_2 + fz_3 + fz_4 = 0
fu = np.array([[0, 2]])
sm_u = SM(fu)
print ('CASE 1:', sm_u)

#case 2) fz_1 + fz_2 + fz_3 + fz_4 = 1
fu = np.array([[1, 1]])
sm_u = SM(fu)
print ('CASE 2:', sm_u)

#case3) fz_1 + fz_2 + fz_3 + fz_4 = 3
fu = np.array([[3, -1]])
sm_u = SM(fu)
print ('CASE 3:', sm_u)









    
    
    








    
    
