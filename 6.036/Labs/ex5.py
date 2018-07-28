import numpy as np

#1) finding X => rank 1

U = np.array([[6, 0, 3, 6]]).T # nx1 - a single regression weight per user
V = np.array([[4, 2, 1]]).T #mx1 - a single feature per movie

X = np.dot(U, V.T)
print (X)


#2) Squared- error estimate

#subbed X's values on entries where Y was undetermined.
Y = np.array([[5, 12, 7], [0, 2, 0], [4, 6, 3], [24, 3, 6]])
squared_error = (Y - X)**2
print (np.sum(squared_error)) #511

#3) total regularization term
lam = 1
reg = (lam/2)*np.sum(X**2)
print (reg)
