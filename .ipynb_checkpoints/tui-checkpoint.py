import pandas
from wordcloud import WordCloud
import matplotlib.pyplot as plt

import model as m

def run():
    class RunScope:
        rawComments = [] #list of strings, each string is a comment
        labelledComments = [] #list of strings, each string is label and associated comment
        posLabelled = [] #list of strings, each string is a positive-labeled comment
        negLabelled = [] #list of strings, each string is a negative-labeled comment
    
    def rdxl(path):
        #C:\Users\n1555085\Downloads\May_Help_Hub_Data.xlsx
        sheet = pandas.read_excel(path)
        courtesy = sheet['Unnamed: 9'].dropna().values.tolist()[1:]
        effectiveness = sheet['Unnamed: 10'].dropna().values.tolist()[1:]
        timeliness = sheet['Unnamed: 11'].dropna().values.tolist()[1:]
        understanding = sheet['Unnamed: 12'].dropna().values.tolist()[1:]
        nps = sheet['Unnamed: 13'].dropna().values.tolist()[1:]
        comments = courtesy + effectiveness + timeliness + understanding + nps
        date = sheet['Unnamed: 1'].dropna().values.tolist()[1:]
        #should add completion rate
        print(f'# of courtesy comments: {len(courtesy)}')
        print(f'# of effectiveness comments: {len(effectiveness)}')
        print(f'# of timeliness comments: {len(timeliness)}')
        print(f'# of understanding comments: {len(understanding)}')
        print(f'# of nps comments: {len(nps)}')
        print(f'# of comments: {len(comments)}')
        RunScope.rawComments = comments
    
    def clsfy(comments):
        RunScope.labelledComments = m.SVMclassifyComments(comments, 'rbf')
        RunScope.posLabelled = []
        RunScope.negLabelled = []
        #lc is (<comment>, (<pos/neg>, <confidence>))
        for lc in RunScope.labelledComments:
            if lc[1][0] == 'pos':
                RunScope.posLabelled.append(lc[0])
            elif lc[1][0] == 'neg':
                RunScope.negLabelled.append(lc[0])
        #remove stopwords
        RunScope.posLabelled = list(filter(lambda s: s , list(map(m.removeStopwords, RunScope.posLabelled))))
        RunScope.negLabelled = list(filter(lambda s: s , list(map(m.removeStopwords, RunScope.negLabelled))))
        m.printLabels(RunScope.labelledComments, len(RunScope.posLabelled), len(RunScope.negLabelled))
        
    def clsfycmt(comment):
        classifiedComment = m.SVMclassify(comment, 'rbf')
        print(f'{classifiedComment[0].upper()} {round(classifiedComment[1], 3)}')
    
    def wc(comments):
        cloud = WordCloud(width = 2000, height = 2000, background_color="white", min_font_size = 30, max_words = 30)
        cloud.generate_from_frequencies(m.wordFrequencies(comments))
        plt.axis("off")
        plt.figure(figsize=(10, 10))
        plt.imshow(cloud, interpolation="bilinear")
        plt.show()
        
        
    def kw(words):
        freq = m.wordFrequencies(words)
        #<word: frequency>
        #for f in freq.values():
        print(freq)

    def help():
        toPrint = "CSAT Comment Analyzer Help:"
        toPrint += "\n    rdxl <file path> - read all comments from excel sheet"
        toPrint += "\n    clsfy - classify all comments from excel sheet"
        toPrint += "\n    clsfycmt <comment> - classify a specific comment"
        toPrint += "\n    wcp - generate wordcloud of positive-labeled comments"
        toPrint += "\n    wcn - generate wordcloud of negative-labeled comments"
        #toPrint += "\n    kwp - find keywords of positive-labeled comments"
        #toPrint += "\n    kwn - find keywords of negative-labeled comments"
        toPrint += "\n    help - print this help menu"
        toPrint += "\n    about - print detailed description of this program and capabilities"
        toPrint += "\n    quit - quit program"
        print(toPrint)
    
    def about():
        toPrint = 'About CSAT Survey Analyzer'
        #more stuff here
        print(toPrint)
        
        
    def numArgs(splitInput):
        return len(splitInput) - 1
    def wrongNumArgsMessage(commandName, expectedNum, numArgs):
        return f'The command \"{commandName}\" takes in {expectedNum} argument(s), you put in {numArgs}. Please type again carefully'
    
    
    
    print("Welcome to Customer Satisfaction Comment Analyzer\n\n")
    help(); print('\n\n>')
    userInput = input()
    while userInput != "quit":
        splitInput = userInput.split()
        if len(splitInput) > 0:
            if splitInput[0] == "rdxl": #<file path>
                if numArgs(splitInput) != 1:
                    print(wrongNumArgsMessage("rdxl", 1, numArgs(splitInput))) 
                    print('Please make sure file path has no spaces')
                    print("\n> ")
                    userInput = input()
                    continue
                rdxl(splitInput[1])
            elif splitInput[0] == "clsfy": #no argument
                if numArgs(splitInput) != 0:
                    print(wrongNumArgsMessage("clsfy", 0, numArgs(splitInput))) 
                    print("\n> ")
                    userInput = input()
                    continue
                clsfy(RunScope.rawComments)
                #print(RunScope.posLabelled)
                #print(RunScope.negLabelled)
            elif splitInput[0] == "clsfycmt": #<comment>, splitInput can be however long >= 2
                if numArgs(splitInput) == 0:
                    print(wrongNumArgsMessage("clsfycmt", 1, 0)) 
                    print("\n> ")
                    userInput = input()
                    continue
                clsfycmt(' '.join(splitInput[1:]))
            elif splitInput[0] == "wcp": #no argument
                if numArgs(splitInput) != 0:
                    print(wrongNumArgsMessage("wcp", 0, numArgs(splitInput))) 
                    print("\n> ")
                    userInput = input()
                    continue
                wc(RunScope.posLabelled)
            elif splitInput[0] == "wcn": #no argument
                if numArgs(splitInput) != 0:
                    print(wrongNumArgsMessage("wcn", 0, numArgs(splitInput))) 
                    print("\n> ")
                    userInput = input()
                    continue
                wc(RunScope.negLabelled)
            elif splitInput[0] == "kwp": #no argument
                if numArgs(splitInput) != 0:
                    print(wrongNumArgsMessage("kwp", 0, numArgs(splitInput))) 
                    print("\n> ")
                    userInput = input()
                    continue
                kw(RunScope.posLabelled)
            elif splitInput[0] == "kwn": #no argument
                if numArgs(splitInput) != 0:
                    print(wrongNumArgsMessage("kwn", 0, numArgs(splitInput))) 
                    print("\n> ")
                    userInput = input()
                    continue
                kw(RunScope.negLabelled) 
            elif splitInput[0] == "help": #no argument
                if numArgs(splitInput) != 0:
                    print(wrongNumArgsMessage("help", 0, numArgs(splitInput))) 
                    print("\n> ")
                    userInput = input()
                    continue
                help()
            elif splitInput[0] == "about": #no argument
                if numArgs(splitInput) != 0:
                    print(wrongNumArgsMessage("about", 0, numArgs(splitInput))) 
                    print("\n> ")
                    userInput = input()
                    continue
                about()
            else: print(f'{splitInput[0]} is an invalid command. Please type again carefully\n')
        print("\n\n> ")
        userInput = input()
    print("...Program exited")
    return

run() #C:\Users\n1555085\Downloads\May_Help_Hub_Data.xlsx












