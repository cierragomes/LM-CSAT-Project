import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import numpy as np
from sklearn.cluster import AgglomerativeClustering

def featureVector(reviewSplit):
    reviewWords = set(reviewSplit)
    features = {}
    for word in vocabulary:
        features[word] = word in reviewWords
    return features

testComment = ['Spenser Cameron set up me and assisted with most of my IT issues. He was exceptional in trying to assist a newcomer on new systems.']

x = featureVector(testComment)




'''
Unsupervised Learning NOTES

Hierarchical - https://stackabuse.com/hierarchical-clustering-with-python-and-scikit-learn (scikit learn)
import matplotlib.pyplot as plt
import pandas as pd
%matplotlib inline
import numpy as np
from sklearn.cluster import AgglomerativeClustering


KMeans


Principle Components Analysis - https://towardsdatascience.com/pca-using-python-scikit-learn-e653f8989e60
keeps principle components, throws away the rest.
could use for training data or model output


Association - https://www.kaggle.com/sangwookchn/association-rule-learning-with-scikit-learn



KNN - https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/ 




























'''