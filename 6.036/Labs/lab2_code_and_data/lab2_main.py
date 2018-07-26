import pdb
import numpy as np
import code_for_lab2 as l2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
import csv
from lab1 import *


#-------------------------------------------------------------------------------
# Auto Data
#-------------------------------------------------------------------------------

# Returns a list of dictionaries.  Keys are the column names, including mpg.
auto_data_all = l2.load_auto_data('auto-mpg.tsv')

# The choice of feature processing for each feature, mpg is always raw and
# does not need to be specified.  Other choices are l2.standard and l2.one_hot.
# 'name' is not numeric and would need a different encoding.
features = [('cylinders', l2.one_hot),
            ('displacement', l2.standard),
            ('horsepower', l2.standard),
            ('weight', l2.standard),
            ('acceleration', l2.standard),
            ('model_year', l2.one_hot),
            ('origin', l2.one_hot)]

features_raw = [('cylinders', l2.one_hot),
            ('displacement', l2.standard),
            ('horsepower', l2.standard),
            ('weight', l2.standard),
            ('acceleration', l2.standard),
            ('model_year', l2.one_hot),
            ('origin', l2.one_hot)]

features_fav = [('cylinders', l2.one_hot),
                ('horsepower', l2.standard),
                ('displacement', l2.standard),
                ('origin', l2.one_hot)]
# Construct the standard data and label arrays
auto_data, auto_labels = l2.auto_data_and_labels(auto_data_all, features_fav)
print('auto data and labels shape', auto_data.shape, auto_labels.shape)

#-------------------------------------------------------------------------------
# Analyze auto data
#-------------------------------------------------------------------------------

# Your code here to process the auto data

#1. I standardized all the continuous valued features and used one-hot encoding for the discrete variables
#2. I used k-fold cross-validation to evaluate the classifiers and scored them starting at T=1,
# if the improvement from the previous T was over epsilon, I kept increasing T by 10 rounds.

Ts = [1,50, 31]
scores_per = []
scores_avg = []
for t in Ts:
    scores_avg.append(xval_learning_alg(averaged_perceptron, auto_data, auto_labels, 10, {'T': t}))
    scores_per.append(xval_learning_alg(perceptron, auto_data, auto_labels, 10, {'T': t}))
print (scores_per) #perceptron: [0.8647279549718576, 0.8748592870544092, 0.9029393370856788]
print (scores_avg) #averaged_perceptron: [0.9158849280800501, 0.9234521575984991, 0.9208880550343966]

##
##start = 1
##epsilon = 0.001
##score_per = xval_learning_alg(perceptron, auto_data, auto_labels, 10, {'T': start})
##while True:
##    start+=10
##    new_score = xval_learning_alg(perceptron, auto_data, auto_labels, 10, {'T': start})
##    if new_score - score_per >= epsilon:
##        score_per = new_score   
##    else:
##        start-=10
##        break
##print ('PERCEPTRON_SCORE', start,score_per)


    
#2.2. Features






#-------------------------------------------------------------------------------
# Review Data
#-------------------------------------------------------------------------------

# Returns lists of dictionaries.  Keys are the column names, 'sentiment' and 'text'.
# The train data has 10,000 examples
review_data = l2.load_review_data('reviews.tsv')
submit_data = l2.load_review_data('reviews_submit.tsv')

# Lists texts of reviews and list of labels (1 or -1)
review_texts, review_label_list = zip(*((sample['text'], sample['sentiment']) for sample in review_data))
submit_texts = [sample['text'] for sample in submit_data]

# The dictionary of all the words for "bag of words"
dictionary = l2.bag_of_words(review_texts)

# The standard data arrays for the bag of words
review_bow_data = l2.extract_bow_feature_vectors(review_texts, dictionary)
review_labels = l2.rv(review_label_list)
print('review_bow_data and labels shape', review_bow_data.shape, review_labels.shape)
# We don't have labels for submit data set
submit_bow_data = l2.extract_bow_feature_vectors(submit_texts, dictionary)

#-------------------------------------------------------------------------------
# Analyze review data
#-------------------------------------------------------------------------------

# Your code here to process the review data


##-----------------------------LAB2--------------------------


#1) FEATURE ENGINEERING

#1.1. Auto-mpg

##pd.set_option('display.max_columns', 10)
##auto_mpg = pd.read_table('auto-mpg.tsv')
##print (auto_mpg.describe())
##auto_mpg['displacement'] = auto_mpg['displacement'].\
##                           apply(12.standard, args = (388.348214, 302.431814))
##auto_mpg['horsepower'] = auto_mpg['horsepower'].\
##                           apply(12.standard, args = (509.354592, 334.078508))
##auto_mpg['weight'] = auto_mpg['weight'].\
##                           apply(12.standard, args = (2977.584184, 849.402560))
##auto_mpg['acceleration'] = auto_mpg['acceleration'].\
##                           apply(12.standard, args = (15.541327, 2.758864))
##
##print (auto_mpg.describe())



##              mpg   cylinders  displacement  horsepower       weight  \
##count  392.000000  392.000000    392.000000  392.000000   392.000000   
##mean     0.000000    5.471939    388.348214  509.354592  2977.584184   
##std      1.001278    1.705783    302.431814  334.078508   849.402560   
##min     -1.000000    3.000000     97.500000  100.000000  1613.000000   
##25%     -1.000000    4.000000    145.750000  147.250000  2225.250000   
##50%      0.000000    4.000000    260.000000  650.000000  2803.500000   
##75%      1.000000    8.000000    443.500000  840.000000  3614.750000   
##max      1.000000    8.000000    980.000000  980.000000  5140.000000   
##
##       acceleration  model_year      origin  
##count    392.000000  392.000000  392.000000  
##mean      15.541327   75.979592    1.576531  
##std        2.758864    3.683737    0.805518  
##min        8.000000   70.000000    1.000000  
##25%       13.775000   73.000000    1.000000  
##50%       15.500000   76.000000    1.000000  
##75%       17.025000   79.000000    2.000000  
##max       24.800000   82.000000    3.000000


##1. mpg:           continuous (modified to be -1 for mpg < 23, 1 for others)
##2. cylinders:     multi-valued discrete [Range (4-8)] [0-4]
##3. displacement:  continuous 
##4. horsepower:    continuous
##5. weight:        continuous
##6. acceleration:  continuous
##7. model year:    multi-valued discrete [Range (70 - 82)][0-12]
##8. origin:        multi-valued discrete [Range (1-3)]
##9. car name:      string (many values)

#standardize the continuous variables, one hot encode the discrete variables

#reason is that if the data points distribution is heavily skewed, then the
#bound R will get too large and perceptron can't converge faster. With
#standardization, the variabes will have zero mean and sd of one.?

#with discrete variables, if we leave them as is, then perceptron will percieve
#a >/< ordering for the diffierent numeric values they could take. one hot
#encoding removes such bias since the weight of these variables is one.
#it is also expands the data points into a higher dimension which might be
#desirable because they may not be linearly separable in a lower feature space


#strategy for evaluating feature vectors

#do k-fold cross validation: with k= 5, #approx = 320 tr_pts and 80 test_pts
##for each choice of feature vectors:
##    run perceptron with num_trials = 2n? and score the classifier
##    record score
##    while true:
##        num_trials += 100
##        run perceptron(num_trials)
##        if score(new_perceptron) has improved by 0.1/0.05?:
##            score = new_score
##            keep going
##        else:
##            break
##    record final score #this is the best perceptron can do with the given
##                        #feature vector
##use the feature vector that had the best score


#1.2. Food reviews

##reviews = pd.read_table('reviews.tsv')
##print (reviews.describe())


#use a bag of words model with puctuation marks and common english words
#stripped. the each unique word will be a feature and its frequency in the
#review will be the value. then once this is complete we can standardize
#the matrix and feed it to perceptron.


#since there's 10,000 inputs, k = 10. the rest, same as before.

#2) EVALUATING ALGORITHMIC AND FEATURE CHOICES FOR AUTO DATA

for feat in range(auto_data.shape[0]):
    print('Feature', feat, features_raw[feat][0])
 # Plot histograms in two windows, note scales!
    fig,(a1,a2) = plt.subplots(nrows=2)
    a1.hist(auto_data[feat,auto_labels[0,:] > 0])
    a2.hist(auto_data[feat,auto_labels[0,:] < 0])
    plt.show()





