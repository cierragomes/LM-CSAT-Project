CSAT Sentiment Analysis Program:

The purpose of this program is to automate the analysis of thousands of customer-written comments collected 
from the surveys that employees take after receiving service from a tech support agent.

This program is composed of numerous built in Python functions as well as written ones that work 
together to perform sentiment analysis tasks. The program uses a natural language processing machine 
called a Support Vector Machine, which is trained by a robust set of prelabelled positive and 
negative data in order to label and display future data as either positive or negative from the 
Tech Support Customer Satisfaction Surveys.

Once the user passes in a file path to a dataset, the text user interface file, test.ipynb, can 
perform different actions to analyze the comments based on the commands passed to it. These commands 
include:
    Reading and collecting comments from an excel sheet that the user passes as a file path
    Classifying each comment as positive or negative along with the confidence score of that classification
    Classifying a specific comment that the user types into the entry spot
    Generating a wordcloud featuring the most important words in the positive labelled comments
    Generating a wordcloud featuring the most important words in the negative labelled comments
    Printing out a help menu describing each command
    Printing a description of the program and its capabilities


In order to run the program, you will need to:
 - download anaconda
 - download nltk libraries
 - install wordcloud on the anaconda command line
 - change any necessary paths
 
In order to run the program, you will need to:
 - restart kernel
 - run all cells in test.ipynb
 