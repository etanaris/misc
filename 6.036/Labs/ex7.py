import numpy as np
#EX7: Neural nets

#1)Predict with steps
X = np.array([[1, 1, 1],\
              [0, 1, 2],\
              [0, 1, 2]])

Y = np.array([[0, 1, 0]]) 

weights = np.array([[-0.5, 1, 0],[1.5, -1, 0]])

def step_actvn(inpt): #derp...must be a better way to do this
    out = []
    for r in inpt:
        inter = []
        for c in r:
            inter.append(1 if c > 0 else 0)
        out.append(inter)
    return np.array(out)

def linear_weights(X, w):
    return np.dot(w, X)

z = linear_weights(X, weights)
f_z = np.vstack((np.ones((1,3)),step_actvn(z)))
vs = np.dot(Y, np.linalg.inv(f_z))
print (z,f_z, vs)

#2) Learn

x_2 = np.array([[1,1,2]]).T
y = np.array([[-1]])
vs_2 = np.array([[1,1,1]])
z_2 = linear_weights(x_2, vs_2)
grad_vs = np.array([[1, 1, 2]])
eta = 0.5
new_vs = vs_2 - (eta*grad_vs)
new_z2 = linear_weights(x_2, new_vs)
newer_vs = new_vs - (eta*grad_vs)
newer_z2 = linear_weights(x_2, newer_vs)
newest_vs = newer_z2 - (eta*grad_vs)
print (z_2, new_vs, new_z2, newer_vs, newer_z2, newest_vs)






