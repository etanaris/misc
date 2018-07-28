import numpy as np

##K nearest neighbors collaborative filtering

#both a and b are row vectors : 1xn
#returns a value between -1 and +1 indicating the degree of correlation between
#variables a and b
def cor(a,b):
    numerator = np.sum(np.multiply(a - np.mean(a), b - np.mean(b)))
    denominator = np.sqrt(np.sum((a - np.mean(a))**2)) *\
                  np.sqrt(np.sum((b - np.mean(b))**2))
    return numerator/denominator

#--------- TEST ------------
# a and b represents ratings that users a and b gave to movies they have both
# watched. the correlation tells you how much alike the users are in their
# movie preference
 
a = np.array([[5, 3]])
b = np.array([[4, 2]])
##print (cor(a, b))


#want to find a prediction for what user 'a' would rate a particular movie 'i'

#by applying the cor function to the pool of other users in the platform, we
#find the top K users who have the highest correlation to 'a' and have watched
#the movie i as well.

#mean_a = the average rating of user a over all movies he's ever watched and rated
#       = this would be the rating you'd predict for 'i' if you had no other info
#       = to rely on
#rating_k = a row vector of the ratings the top K users most correlated to user 'a' gave
#       = to movie 'i'

#k_mean = a row vector of the overall average rating of the k users. this is
#       = to work with deviations from the mean and remove any bias the users
#       = may have when rating movies in general.

#sim_ak = a row vector of cor(a,k)
        
def predict(mean_a, rating_k, mean_k, sim_ak):
    weighed_sum = np.sum(np.multiply(sim_ak, rating_k - mean_k))
    normalized_sum = weighed_sum/np.sum(abs(sim_ak))
    return mean_a + normalized_sum


    


                  
                  
    
    
