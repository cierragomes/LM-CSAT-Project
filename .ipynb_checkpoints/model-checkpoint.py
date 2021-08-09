import pandas
from nltk.corpus import stopwords
from string import punctuation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC

from sklearn.cluster import KMeans
import numpy as np
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.probability import FreqDist
from collections import defaultdict
from heapq import nlargest


#TRAINING DATA
with open(r'positiveComments.txt', 'r') as f:
    posReviews = f.readlines()
with open(r'negativeComments.txt', 'r') as f:
    negReviews = f.readlines()

sw = set(stopwords.words('english') + list(punctuation))
notStopwords = ['not', 'no', '!', 'but', 'too', 'have', 'had']

def removeStopwords(review):
    #review.translate(None, string.punctuation)
    return ' '.join([word for word in review.split() if word.lower() not in sw or word.lower() in notStopwords])
posReviews = list(filter(lambda s: s , list(map(removeStopwords, posReviews))))
negReviews = list(filter(lambda s: s , list(map(removeStopwords, negReviews))))
posWords = [word.lower() for review in posReviews for word in review.split()]
negWords = [word.lower() for review in negReviews for word in review.split()]
vocabulary = list(set(posWords + negWords))


#SVM CLASSIFIER TRAINING
#vectorizer = TfidfVectorizer(min_df = 5, max_df = 0.8, sublinear_tf = True, use_idf = True, ngram_range=(1, 2))
vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
train_vectors = vectorizer.fit_transform(posReviews + negReviews)
labelsList = ['pos'] * len(posReviews) + ['neg'] * len(negReviews)

classifier_linear = SVC(kernel='linear', probability = True)
classifier_linear.fit(train_vectors, labelsList)

classifier_poly = SVC(kernel='poly', probability = True)
classifier_poly.fit(train_vectors, labelsList)

classifier_rbf = SVC(kernel='rbf', probability = True)
classifier_rbf.fit(train_vectors, labelsList)

classifier_sigmoid = SVC(kernel='sigmoid', probability = True)
classifier_sigmoid.fit(train_vectors, labelsList)

# svm kernel can be ‘linear’, ‘poly’, ‘rbf’, or ‘sigmoid’
def SVMclassify(review, kernel):
    review = removeStopwords(review)
    review_vector = vectorizer.transform([review]) # vectorizing
    if kernel == 'linear':
        return (classifier_linear.predict(review_vector)[0], max(classifier_rbf.predict_proba(review_vector)[0]))
    elif kernel == 'poly':
        return (classifier_poly.predict(review_vector)[0], max(classifier_rbf.predict_proba(review_vector)[0]))
    elif kernel == 'rbf':
        return (classifier_rbf.predict(review_vector)[0], max(classifier_rbf.predict_proba(review_vector)[0]))
    elif kernel == 'sigmoid':
        return (classifier_sigmoid.predict(review_vector)[0], max(classifier_rbf.predict_proba(review_vector)[0]))
    return None

def SVMclassifyComments(comments, kernel):
    return [(c, SVMclassify(c, kernel)) for c in comments]

#param [(<comment>, (<label>, <confidence>))]
def printLabels(labelledComments, lenp, lenn):
    print('OUTPUT OF SUPPORT VECTOR MACHINE CLASSIFIER:')
    print(f'{lenp} positive-labeled comments, {lenn} negative-labeled comments\n\n')
    for lc in labelledComments:
        print(lc[1][0].upper(), round(lc[1][1], 3), ':', f'\"{lc[0]}\"', '\n')
        

#WORDCLOUD STUFF
cloudStopwords = set(stopwords.words('english') + list(punctuation))
ignore = ['resolve', 'resolved', 'resolution', 'issue', 'problem', 'solve', 'ticket', 'request']
cloudStopwords = cloudStopwords.union(set(ignore))

#calculates word frequencies for generating wordcloud
#parameter should be list of comments, returns dictionary
def wordFrequencies(commentList):
    dictionary = {}
    words = ' '.join(commentList).lower().split()
    for w in words:
        if w not in dictionary and w not in cloudStopwords:
            dictionary[w] = words.count(w)
    return dictionary

def customWordFrequencies(commentList, customWords):
    dictionary = {}
    words = ' '.join(commentList).lower().split()
    for w in words:
        if w in customWords and w not in cloudStopwords:
            dictionary[w] = words.count(w)
    return dictionary
