#perceptron algorithm

import numpy as np
import random 

# input ==> a set of training point (x,y) where x is a column feature vector
#           and y is its label. y E {+1, -1}

# outupt ==> a parameter w (column vector) that linearly classifies the training
#           set with zero error.

# assumption: the data is linearly separable in R(d+1)

#normalizes a vector
def normalize(training_set):
    mag = np.linalg.norm(training_set)
    return (1/mag) * training_set

#adds 1 to the feature vector x to account for the offset to the classifying hyperplane
#that would otherwise be anchored at the origin 
def offset_adjusted(training_set):
    return [(np.insert(x, 0, 1), y) for x,y in training_set]


#the error function is the ratio of misclassified training points
def perceptron(training_set):
    #returns the first w that linearly classifies the training set and the
    #number of iterations it took to converge to w
    training_set = normalize(offset_adjusted(training_set))
    mistakes = 0
    w = 0
    eta = 1/len(training_set)   #learning rate..how to optimally choose this to converge faster?
    while len(misclassified_tps(w, training_set)) != 0:
        misclassified = misclassified_tps(w, training_set)
        m_pt = misclassified[random.randint(0, len(misclassified)-1)]
        w = w + eta * (m_pt(0) * m_pt(1))
        mistakes +=1
    return w, mistakes


def misclassified_tps(w, training_set, hinge_loss = False):
    #returns the set of misclassified training points with w in the trianing set
    #hinge_loss == True means we'll consider a training point misclassified if
    #it goes between the gutters at x.w = +/-1
    if hinge_loss:
        return [(x,y) for x,y in training_set if y *(np.dot(w, x)) <= 1]
    else:
        return [(x,y) for x,y in training_set if y *(np.dot(w, x)) <= 0]
 

#theoretical convergence
def convergence(training_set):
    w_opt, mistakes = perceptron(training_set)
    bound = max([np.linalg.norm(x) for x in x,y training_set])
    gamma = min([(y*np.dot(w_opt, x))/np.linalg.norm(w_opt) for x,y in traning_set])
    convergence = (bound/gamma)**2
    return convergence, mistakes <= convergence
    
    
    
        
