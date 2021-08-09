CSAT Sentiment Analysis Program:

The purpose of this program is to automate the analysis of customer-written comments collected 
from the surveys that employees take after receiving service from a tech support agent. This analysis will
assist in creating the monthly Customer Satisfaction Review presentations that the tech support team makes.

This program is composed of numerous built in Python functions as well as written ones that work 
together to perform sentiment analysis tasks. The program uses a natural language processing machine 
called a Support Vector Machine, which is trained by a robust set of prelabelled positive and 
negative data in order to label and display future data as either positive or negative from the 
Tech Support Customer Satisfaction Surveys.


Initial Setup:
 - clone this repository to a local folder
 - download anaconda 3 from LAP
 - download nltk libraries
 - install wordcloud on the anaconda command prompt
 - download xlsx file of survey data, replace spaces with _, and copy path
 
In order to run the program, you will need to:
 - navigate to run.ipynb
 - restart kernel
 - run all cells
 
From here a text-line interface should pop up, with which you can execute commands of the program on:
 - First you will need to do the 'rdxl' command, which takes in a file path as a parameter. This is how the 
   program knows what excel sheet you want to analyze, so you must read a sheet first before doing anything
   else. For example, "rdxl C:\Users\n1555085\Downloads\May_Help_Hub_Data.xlsx"
 - Second, run the 'clsfy' command to classify every comment in the esxcel sheet. You will need to execute this
   command before running 'wcp' or 'wcn'
 - When you are done running the program make sure to do the 'quit' command. If you do not quit and then try to
   restart and run the kernel, you will have to reload Jupyter
 
 
If you run into any problems or have questions, contact jlshen@umass.edu
 