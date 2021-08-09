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
import matplotlib.pyplot as plt

#TRAINING DATA
with open(r'positiveComments.txt', 'r') as f:
    posReviews = f.readlines()
with open(r'negativeComments.txt', 'r') as f:
    negReviews = f.readlines()

def removePunctuation(review):
    return review.translate(str.maketrans('', '', punctuation))
def removeStopwords(review):
    sw = set(stopwords.words('english')) #sw.union(['trouble', 'having'])
    notSW = ['not', 'no', '!', 'but', 'too', 'have', 'had']
    return ' '.join([word for word in review.split() if word.lower() not in sw or word.lower() in notSW])
def normalize(review):
    return removeStopwords(removePunctuation(review)).lower()
def normalizeReviews(reviews):
    return list(map(normalize, reviews))

posReviews = normalizeReviews(posReviews)
negReviews = normalizeReviews(negReviews)
posWords = [word for review in posReviews for word in review.split()]
negWords = [word for review in negReviews for word in review.split()]
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
    fig1, ax1 = plt.subplots()
    ax1.pie([lenp, lenn], explode=(0,0), labels=['Positive', 'Negative'], autopct='%1.1f%%',
    shadow=True, startangle=90)
    ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()

#WORDCLOUD STUFF
cloudStopwords = set(stopwords.words('english') + list(punctuation))
ignore = ['resolve', 'resolved', 'resolution', 'issue', 'problem', 'solve', 'ticket', 'request']
ignore += ['issues', 'response', 'solved', 'could', 'didnt']
ignore += ['1', '2', '3', '4', '5', '6', '7', '8', '9']
ignore += ['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
cloudStopwords = cloudStopwords.union(set(ignore))

#calculates word frequencies for generating wordcloud
#parameter should be list of comments, returns dictionary
def wordFrequencies(commentList):
    dictionary = {}
    words = ' '.join(normalizeReviews(commentList)).lower().split()
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



