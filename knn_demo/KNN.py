#!/usr/bin/env python
# coding: utf-8

# In[2]:



import operator
from numpy import *

# In[1]:


# create a dataset which contains 4 samples with 2 classes
def createDataSet():
    # create a matrix: each row as a sample
    group = array([[1,9],[1,1],[0.1,0.2],[0,0.1]])
    labels = ["A","A","B","B"]
    return group, labels


# In[7]:


# classify using KNN
def KNNClassify(newInput, dataSet, labels, k):
    numSamples = dataSet.shape[0] # shape[0] stands for the num of rows
    
    # step 1: calculate Euclidean distance
    # tile(A,reps): construct an array by repeating A reps times
    # the following copy numSampes rows for dataset
    diff = tile(newInput, (numSamples,1))-dataSet
    squaredDiff = diff ** 2 
    squaredDist = sum(squaredDiff, axis=1)
    distance = squaredDist ** 0.5 # obtain square root
    
    # step 2: sort the distance
    # argsort() returns the indices that would sort an array in a ascending order
    sortedDistIndices = argsort(distance)
    
    classCount={} # define a dictionary (can be append element)
    for i in range(k):
        # step 3: choose the min k distance
        voteLabel = labels[sortedDistIndices[i]]
        
        # step 4: count the times labels occur
        # when the key voteLabel is not in dictionary classCount.get() will return 0
        classCount[voteLabel] = classCount.get(voteLabel,0) +  1
        
    # step 5: the max voted class will return
    maxCount = 0
    for key, value in classCount.items():
        if value > maxCount:
            maxCount = value
            maxIndex = key
            
    return maxIndex
      


# In[6]:





# In[ ]:





# In[ ]:




