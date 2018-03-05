import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
import numpy as np
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
import re
import numpy
import copy
import collections
def tokenize(text):
    tokens=word_tokenize(text)
    sentTokens=sent_tokenize(text)
    print("The sentences are")
    print(sentTokens)
    print("the tokens are")
    print(tokens)
    newSentToken=[re.sub(r'\n',"", i) for i in sentTokens]
    print("the tokens are")
    print(newSentToken)
    print("the Frequency distribution is")
    print(FreqDist(tokens).most_common())
    trimtrx,triforwardDict,tribackwardDict=language_model(tokens,3)
    bimtrx,biforwardDict,bibackwardDict=language_model(tokens,2) 
    unimtrx,uniforwardDict,unibackwardDict =language_model(tokens,1)
    biprobMtrx,triprobMtrx=probability_matrix(unimtrx,uniforwardDict,bimtrx,biforwardDict,trimtrx,triforwardDict)
    ##return(mtrx,forwardDict,backwardDict,unimtrx,uniforwardDict,unibackwardDict)
def language_model(tokenList,gramCount):
    sortedTokens=sorted(tokenList)
    dictList=[]
    ##this dict search word to matrix for creating model
    modelDict={}
    ## this dict for matrix to word search for creating and computing probabilities 
    backwardsDict={}
    count=len(modelDict)
    for i in sortedTokens:
        if i not in modelDict:
            modelDict[i]=count
            backwardsDict[count]=i
            count+=1
    dictList.append(modelDict)
    for i in range(gramCount-1):
        modelDictNextAxis=copy.deepcopy(modelDict)
        dictList.append(modelDictNextAxis)
    ## n dimensional matrix
    firstDimension=[]
    firstDimension= [0 for i in range(len(dictList[0]))]
    print(firstDimension)
    nDimensionList=copy.deepcopy(firstDimension)
    for i in range(gramCount-1):
        holder=copy.deepcopy(nDimensionList)
        nDimensionList=[copy.deepcopy(holder) for i in range(len(holder))]
    print(nDimensionList)
    print(backwardsDict)
    print(modelDict)
    outputMtrx=compute_probability(modelDict,tokenList,nDimensionList,gramCount)
    return outputMtrx,modelDict,backwardsDict
def probability_matrix(unimtrx,uniFor,bimtrx,biFor,trimtrx,triFor):
    probBiMtrx= np.zeros(shape=(len(bimtrx), len(bimtrx)))
    probTriMtrx=np.zeros(shape=(len(trimtrx), len(trimtrx),len(trimtrx)))
    totalbi=0
    totaltri=0
    for i in range(len(bimtrx)):
        for j in range(len(bimtrx[i])):
            probBiMtrx[i][j]=bimtrx[i][j]/(unimtrx[i]+len(unimtrx)*len(unimtrx))
            totalbi+=bimtrx[i][j]/(unimtrx[i]+len(unimtrx)*len(unimtrx))
    
    print(probBiMtrx) 
    for i in range(len(trimtrx)):
        for j in range(len(trimtrx)):
            for k in range(len(trimtrx)):
                probTriMtrx[i][j][k]=trimtrx[i][j][k]/(bimtrx[i][j]+len(bimtrx)*len(bimtrx)*len(bimtrx))
                totaltri+=trimtrx[i][j][k]/(bimtrx[i][j]+len(bimtrx)*len(bimtrx)*len(bimtrx))
    print(probTriMtrx)
    print(totalbi)
    print(totaltri)
    return(probBiMtrx,probTriMtrx)
def compute_probability(modelDict,tokenList,mtrx,gramCount):
    smoothinghelperlength=0
    smoothinghelperList=[]
    for i in range(len(tokenList)-gramCount+1):
        holder=[]
        for j in range(gramCount):
            holder.append(modelDict[tokenList[i+j]])
        print(holder)
        if (gramCount==1):
            unigram(mtrx,holder)
        elif(gramCount==2):
            bigram(mtrx,holder)
        elif(gramCount==3):
            trigram(mtrx,holder)
    print(mtrx)
    smoothinghelper(mtrx,gramCount)
    return mtrx
def smoothinghelper(mtrx,gramCount):
    if(gramCount==1):
        for i in range(len(mtrx)):
            mtrx[i]+=1
    if(gramCount==2):
        for i in range(len(mtrx)):
            for j in range(len(mtrx)):
                mtrx[i][j]+=1
    if(gramCount==3):
        for i in range(len(mtrx)):
            for j in range(len(mtrx)):
                for k in range(len(mtrx)):
                    mtrx[i][j][k]+=1
##Please use the same gram count which you eentered for creation or else this program will craash
def sentence_probability(sentence,mtrx,forwardDict,unimtrx,uniforwardDict,gramCount):
    sentenceTokened=word_tokenizer(sentence)
    probability=1
    if gramCount==2:
        for i in range(len(sentenceTokened)):
            if i ==0:
                ##the plus is for smoothing
                probability=probability*(sentenceTokenized[i]/(len(uniforwardDict)+len(uniforwardDict)))
    ##if gramCount==3:
    
def unigram(mtrx,holder):
    mtrx[holder[0]]+=1
def bigram(mtrx,holder):
    ##remember works from holder[1] given holder[0] and I did not bother change
    ## this way for mtrx[][] so if mtrx[3][5] is 1 this means 5 given one has one count not other way around
    mtrx[holder[0]][holder[1]]+=1
def trigram(mtrx,holder):
    mtrx[holder[0]][holder[1]][holder[2]]+=1
def main():
    nltk.download('punkt')
    documents = reuters.fileids()
    test=reuters.raw("test/15556")
    mtrx,forwardDict,backwardDict,unimtrx,uniforwardDict,unibackwardDict=tokenize(test)
    print(mtrx,forwardDict,backwardDict,unimtrx,uniforwardDict,unibackwardDict)
    
main()
        
