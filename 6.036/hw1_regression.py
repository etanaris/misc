import numpy as np
import sys

lambda_input = int(sys.argv[1])
sigma2_input = float(sys.argv[2])
X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
y_train = np.genfromtxt(sys.argv[4])
X_test = np.genfromtxt(sys.argv[5], delimiter = ",")
n, d = X_train.shape

## Solution for Part 1
def part1():
    ## Input : Arguments to the function
    ## Return : wRR, Final list of values to write in the file
    term1 = lambda_input * np.eye(d) + np.matmul(X_train.T,X_train)
    wRR = np.matmul(np.linalg.inv(term1), np.matmul(X_train.T, y_train))
##    print ('WRR', wRR, 'shape', wRR.shape)
    return wRR

wRR = part1()  # Assuming wRR is returned from the function

np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n") # write output to file


## Solution for Part 2
def part2():
    ## Input : Arguments to the function
    ## Return : active, Final list of values to write in the file
    test_set = dict((i,j) for i,j in zip(range(1, (X_test.shape[0]+1)),[row for row in X_test]))
    term1 = lambda_input * np.eye(d) + (1/sigma2_input) * np.matmul(X_train.T,X_train)
    E = np.linalg.inv(term1) #dxd
    X_new = X_train 
    output = []
    loops = max(10, len(test_set))

    for i in range(loops):
##        if len(test_set) == 0:
##            break
        sigma_max = 0
        x_max = np.array([])
        idx_max = -sys.maxsize
        for idx, x_o in test_set.items():
            #print ('X_0', x_o)
            sigma_o = sigma2_input + np.matmul(x_o, np.matmul(E, x_o))
            if sigma_o > sigma_max:
                sigma_max = sigma_o
                x_max = x_o
                idx_max = idx
##        print ('X_new', X_new, 'X_new shape', X_new.shape)
##        print ('x_max', x_max, 'x_max shape', x_max.shape)
##        print ('sigma_max', sigma_max)
##        print ('test_set', len(test_set))
        X_new= np.vstack([X_new, x_max])      
        term1_new = lambda_input * np.eye(d) + (1/sigma2_input) * np.matmul(X_new.T, X_new)
        E = np.linalg.inv(term1_new)
        output.append(idx_max)
        del test_set[idx_max]
##        print (output)
    return output        

active = part2()  # Assuming active is returned from the function
np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv" , active, delimiter=",") # write output to file
