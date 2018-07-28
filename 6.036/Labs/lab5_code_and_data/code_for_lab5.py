import pdb
import numpy as np

# Data is a list of (i, j, r) triples
ratings_small = \
[(0, 0, 5), (0, 1, 3), (0, 3, 1),
 (1, 0, 4), (1, 3, 1), 
 (2, 0, 1), (2, 1, 1), (2, 3, 5), 
 (3, 0, 1), (3, 3, 4), 
 (4, 1, 1), (4, 2, 5), (4, 3, 4)]

def pred(data, x):
    (a, i, r) = data
    (u, b_u, v, b_v) = x
    return np.dot(u[a].T,v[i]) + b_u[a] + b_v[i]

# Utilities

# Compute the root mean square error
def rmse(data, x):
    error = 0.
    for datum in data:
        error += (datum
                  [-1] - pred(datum, x))**2
    return np.sqrt(error/len(data))

# Counts of users and movies, used to calibrate lambda
def counts(data, index):
    item_count = {}
    for datum in data:
        j = datum[index]
        if j in item_count:
            item_count[j] += 1
        else:
            item_count[j] = 1
    c = np.ones(max(item_count.keys())+1)
    for i,v in item_count.items(): c[i]=v
    return c

# The ALS outer loop
def mf_als(data_train, data_test, k=2, lam=0.02, max_iter=100):
    # size of the problem
    n = max(d[0] for d in data_train)+1 # users
    m = max(d[1] for d in data_train)+1 # items
    # which entries are set in each row and column
    us_from_v = [[] for i in range(m)]  # II (i-index-set)
    vs_from_u = [[] for a in range(n)]  # AI (a-index set)
    for (a, i, r) in data_train:
        us_from_v[i].append((a, r))
        vs_from_u[a].append((i, r))
    # global offset, mean of ratings
    b = sum(r for (a,i,r) in data_train)/len(data_train)
    # Initial guess at u, b_u, v, b_v
    # Note that u and v are lists of column vectors (columns of U, V).
    x = ([np.random.normal(1/k, size=(k,1)) for a in range(n)],
          np.zeros(n),
          [np.random.normal(1/k, size=(k,1)) for i in range(m)],
          np.zeros(m))
    # Alternation, modifies the contents of x
    for i in range(max_iter):
        update_U(data_train, vs_from_u, x, k, lam)
        update_V(data_train, us_from_v, x, k, lam)
    # The root mean square errors measured on test set
    print('rmse=', rmse(data_test, x))
    return x

# X : n x k
# Y : n
def ridge_analytic(X, Y, lam):
    (n, k) = X.shape
    xm = np.mean(X, axis = 0, keepdims = True)   # 1 x n
    ym = np.mean(Y)                              # 1 x 1
    Z = X - xm                                   # d x n
    T = Y - ym                                   # 1 x n
    th = np.linalg.solve(np.dot(Z.T, Z) + lam * np.identity(k), np.dot(Z.T, T))
    # th_0 account for the centering
    th_0 = (ym - np.dot(xm, th))                 # 1 x 1
    return th.reshape((k,1)), float(th_0)

# Example from lab handout
Z = np.array([[1], [1], [5], [1], [5], [5], [1]])
b_v = np.array([[3], [3], [3], [3], [3], [5], [1]])
B = np.array([[1, 10], [1, 10], [10, 1], [1, 10], [10, 1], [5, 5], [5, 5]])
# Solution with offsets, using ridge_analytic provided in code file
u_a, b_u_a = ridge_analytic(B, (Z - b_v), 1)
print('With offsets', u_a, b_u_a)
# Solution using previous model, with no offsets
u_a_no_b = np.dot(np.linalg.inv(np.dot(B.T, B) + 1 * np.identity(2)), np.dot(B.T, Z))
print('With no offsets', u_a_no_b)

##y = u.v + b_u + b_v : prediction model with offset
def update_U(data, vs_from_u, x, k, lam):
    (u, b_u, v, b_v) = x
    ua_updates = [] ; bu_updates  = [] #weight vectors, user offsets
    for j in range(len(vs_from_u)):
        a = vs_from_u[j] #i,r
        if not a: #no movie ratings from user a => no update 
            ua_updates.append(u[j]); bu_updates.append(b_u[j]) 
        else: #ratings from user a
            B = np.vstack(tuple(v[i].T for i,r in a)) #laXk
            Z = np.vstack(tuple(r for i,r in a)) #lax1
            b_va = np.vstack(tuple(b_v[i] for i,r in a)) #laXk
            ua, b_ua = ridge_analytic (B, (Z - b_va), lam)
            ua_updates.append(ua); bu_updates.append(b_ua)
    x = (ua_updates, bu_updates, v, b_v)
    return x

def update_V(data, us_from_v, x, k, lam):
    (u, b_u, v, b_v) = x
    vi_updates = [] ; bv_updates  = [] #movie feature vectors, movie offsets
    for j in range(len(us_from_v)):
        m = us_from_v[j] #a,r
        if not m: #no user has seen movie => no update 
            vi_updates.append(v[j]); bv_updates.append(b_v[j]) 
        else: #some users have seen it
            B = np.vstack(tuple(u[a].T for a,r in m)) #laXk
            Z = np.vstack(tuple(r for a,r in m)) #lax1
            b_ua = np.vstack(tuple(b_u[a] for a,r in m)) #laXk
            vi, b_vi = ridge_analytic (B, (Z - b_ua), lam)
            vi_updates.append(vi); bv_updates.append(b_vi)
    x = (u, b_u, vi_updates, bv_updates)
    return x


# Simple test case
print("ALS")
mf_als(ratings_small, ratings_small,
       lam=0.01, max_iter=10, k=2)

# The SGD outer loop
def mf_sgd(data_train, data_test, step_size_fn, k=2, lam=0.02, max_iter=100):
    # size of the problem
    ndata = len(data_train)
    n = max(d[0] for d in data_train)+1
    m = max(d[1] for d in data_train)+1
    # Distribute the lambda among the users and items
    lam_uv = lam/counts(data_train,0), lam/counts(data_train,1)
    # Initial guess at u, b_u, v, b_v (also b)
    x = ([np.random.normal(1/k, size=(k,1)) for j in range(n)],
         np.zeros(n),
         [np.random.normal(1/k, size=(k,1)) for j in range(m)],
         np.zeros(m))
    di = int(max_iter/10.)
    for i in range(max_iter):
        if i%di == 0:
            print('i=', i, 'rmse=', rmse(data_test, x))
        step = step_size_fn(i)
        j = np.random.randint(ndata)            # pick data item
        sgd_step(data_train[j], x, lam_uv, step) # modify x
    print('k=', k, 'rmse', rmse(data_test, x))

def sgd_step(data, x, lam, step): #step is learning rate
    (a, i, r) = data #user_idx, movie_idx, rating
    (u, b_u, v, b_v) = x
##    u[a], v[i] #kx1, kx1
    (lam_u, lam_v) = lam
    div =  np.dot(v[i].T, u[a]) - r #deviation of the prediction from the real rating
##    print ('Div', div, 'vi', v[i])
    grad_u = div * v[i] + lam_u[a]*np.linalg.norm(u[a])
    grad_v = div * u[a] + lam_v[i]*np.linalg.norm(v[i])
    u[a] -= (step*grad_u); v[i] -= (step*grad_v)
    b_u[a]-= (step*div) ; b_v[i] -= (step*div)
    return x

# Simple test case
print("SGD")
mf_sgd(ratings_small, ratings_small, step_size_fn=lambda i: 0.1,
       lam=0.01, max_iter=1000, k=2)

import csv
def load_ratings_data(path_data):
    """
    Returns a list of triples (i, j, r)
    """
    fields = ['user_id', 'item_id', 'rating']
    data = []
    with open(path_data) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            data.append([int(datum[f]) for f in fields])
    print('Loading from', path_data, 
          'users', len(set(x[0] for x in data)), 
          'items', len(set(x[1] for x in data)))
    return data

# Load the movie data
dir = ''
b1 = load_ratings_data(dir+'ml_100k_u1_base.tsv') # train
t1 = load_ratings_data(dir+'ml_100k_u1_test.tsv') # test

def load_movies(path_movies):
    """
    Returns a dictionary mapping item_id to item_name
    """
    fields = ['item_id', 'item_name']
    data = {}
    with open(path_movies) as f_data:
        for datum in csv.DictReader(f_data, delimiter='\t'):
            data[int(datum['item_id'])] = datum['item_name']
    return data
#all_movie_names = load_movies(dir+'movies1.tsv')

def baseline(train, test):
    item_sum = {}
    item_count = {}
    total = 0
    for (i, j, r) in train:
        total += r
        if j in item_sum:
            item_sum[j] += 3
            item_count[j] += 1
        else:
            item_sum[j] = r
            item_count[j] = 1
    error = 0
    avg = total/len(train)
    for (i, j, r) in test:
        pred = item_sum[j]/item_count[j] if j in item_count else avg
        error += (r - pred)**2
    return np.sqrt(error/len(test))

print('Baseline rmse (predict item average)', baseline(b1, t1))
print('Running on the MovieLens data')
for k in (1, 2, 3):
    print('ALS, k=', k)
    mf_als(b1, t1, lam = 1, max_iter=20, k=k)
    print('SGD, k=', k)
    mf_sgd(b1, t1, lam = 1, step_size_fn=lambda i: 0.01, max_iter=500000, k=k)


##-------------------------------LAB 5 WRITEUP-----------------------------
##1) Checkoff on recommender systems
## 1.1) Complete data

#1) It can't be because the it's a rank 2 matrix

#2) It can. Y*I = U(V.T)

## 1.2) Two kinds people, two kinds of movies

## 1.3) Half-fixed

## done in the notes. finding the optimal U_a, regression weights for some
##particular user
    
#15) pats optimal regression weights for each feature reperesenting a movie

##V_i is the 'feature vector' for some movie
B = np.array([[1,10],[1,10],[10,1], [1, 10], [10,1]]) #V_is pat has rated LxK
Z = np.array([[1],[1],[5],[1],[5]]) #pat's real rating
lam = 1
ua_pat = np.dot(np.linalg.inv(np.dot(B.T,B)+ lam*np.eye(B.shape[1])),\
                np.dot(B.T, Z))
##print ('pats regression weights:', ua_pat) #Kx1: [[0.49246358],[0.05074858]]

#16)pat's predicted rating
pats_rating= np.dot(B,ua_pat)
##print ('predicted rating of training points:', pats_rating,\
##       'real rating of training points:', Z)
test_point = np.array([[10,1]]) #llama movie vector
test_point_r = np.array([[1, 10]]) #robot movie vector
pats_llama_rating = np.dot(test_point, ua_pat)
##print ('predicted rating of test point:', pats_llama_rating) #4.9753844

## 1.4) Some movies are more equal than others
##17) pat's feeling about a new lamma movie, with/without offset
ua_pat_no_offset = np.array([[ 0.50148126],[ 0.0562376 ]])
ua_pat_offset = np.array([[ 0.22024566],[-0.22193986]])
b_ua_pat = np.array([[ 0.00762389]])
b_v_llama = np.array([[1]]) #everybody hated it
b_v_robot = np.array([[3]]) #average offset#([[5]]) #everybody loved it
pats_feels_offset= float(np.dot(test_point, ua_pat_offset) + b_ua_pat + b_v_llama)
pats_feels_no_offset = float(np.dot(test_point, ua_pat_no_offset))
pats_feels_offset_r= float(np.dot(test_point_r, ua_pat_offset) + b_ua_pat + b_v_robot)
pats_feels_no_offset_r = float(np.dot(test_point_r, ua_pat_no_offset))
##print ('llama:', [pats_feels_no_offset, pats_feels_offset])
##print ('robot:', [pats_feels_no_offset_r, pats_feels_offset_r])


#2) IMPLEMENTING RECOMMENDER SYSTEMS

##2.3) MovieLens











    
