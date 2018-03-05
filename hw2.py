import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.probability import FreqDist
from nltk.corpus import inaugural
import math
import random
import re
import numpy
import copy
import collections
def tokenize(text):
    tokens=word_tokenize(text)
    sentTokens=sent_tokenize(text)
    print("\nThe sentences are")
    print(sentTokens)
    print("\nthe tokens are")
    print(tokens)
    newSentToken=[re.sub(r'\n',"", i) for i in sentTokens]
    print("\nthe tokens are")
    print(newSentToken)
    print("\nthe Frequency distribution is")
    freqDict=dict(FreqDist(tokens).most_common())
    freqDict["UNK"]=1
    print(freqDict)
    print(FreqDist(tokens))
    forwardDict,backwardsDict,probMatrix,probUniMatrix,totalProb=language_model(tokens,freqDict)
    return forwardDict,backwardsDict,probMatrix,probUniMatrix,totalProb
def language_model(tokenList,freqDist):
    sortedTokens=sorted(tokenList)
    print("the osrrted toens")
    sortedTokens.append("UNK")
    print(sortedTokens)
    forwardDict={}
    backwardsDict={}
    count=len(forwardDict)
    for i in sortedTokens:
        if i not in forwardDict:
            forwardDict[i]=count
            backwardsDict[count]=i
            count+=1
    print("\nthe forward dictionary")
    print(forwardDict)
    print("\nthe backward dictionary")
    print(backwardsDict)
    modelMatrix=numpy.zeros((len(forwardDict),len(forwardDict)))
    probMatrix=numpy.zeros((len(forwardDict),len(forwardDict)))
    probUniMatrix=numpy.zeros((len(forwardDict)))
    print("\nbefore any values added to language model matrix")
    print(modelMatrix)
    totalProb=compute_probability(forwardDict,tokenList,modelMatrix,freqDist,backwardsDict,probMatrix,probUniMatrix)
    print(probMatrix)
    print(modelMatrix)
    print(len(tokenList))
    print(probUniMatrix)
    return(forwardDict,backwardsDict,probMatrix,probUniMatrix,totalProb)
    
def compute_probability(modelDict,tokenList,mtrx,freqDist,backwardsDict,probMtrx,uniProbMatrix):
    totalProb=0
    totalUni=0
    for i in range(len(tokenList)):
        if (i<len(tokenList)-1):
            indexFirstSeg=modelDict[tokenList[i]]
            indexSecondSeg=modelDict[tokenList[i+1]] 
            mtrx[indexFirstSeg,indexSecondSeg]+=1
    for i in range(len(mtrx)):
        for j in range(len(mtrx[i])):
            mtrx[i,j]+=1
    print("postadd")
    print(mtrx)
    for i in range(len(mtrx)):
        uniProbMatrix[i]=freqDist[backwardsDict[i]]/(len(tokenList)+1)
        totalUni+=uniProbMatrix[i]      
        for j in range(len(mtrx[i])):
            ##probMtrx[i,j]=(mtrx[i,j]/((freqDist[backwardsDict[i]])+len(freqDist)))*(freqDist[backwardsDict[i]]/(len(tokenList)+1))
            totalProb+=(mtrx[i,j]/((freqDist[backwardsDict[i]])+len(freqDist)))*(freqDist[backwardsDict[i]]/(len(tokenList)+1))
            probMtrx[i,j]=(mtrx[i,j]/((freqDist[backwardsDict[i]])+len(freqDist)))
            #totalProb+=(mtrx[i,j]/((freqDist[backwardsDict[i]])+len(freqDist)))
    print("after smooth")
    print(totalUni)
    return(totalProb)
 
def sentence_probability(sentence,probMtrx,forwardDict,probUniMatrix):
    probability=1
    for i in range(len(sentence)-1):
        wordi=sentence[i]
        wordj=sentence[i+1]
        if wordi not in forwardDict:
            wordi="UNK"
        if wordj not in forwardDict:
            wordj="UNK"
        if i==0:
            print(probability)
            print(probUniMatrix[forwardDict[wordi]])
            probability= probability*probUniMatrix[forwardDict[wordi]]
        else:
            print(probability)
            print(probMtrx[forwardDict[wordi]][forwardDict[wordj]])
            probMtrx[forwardDict[wordi]][forwardDict[wordj]]
            probability=probability*probMtrx[forwardDict[wordi]][forwardDict[wordj]]
    return(probability)
def sentence_perplex(sentence,probMtrx,forwardDict,probUniMatrix):
    sent_token=word_tokenize(sentence)
    probability=1
    for i in range(len(sent_token)-1):
        wordi=sent_token[i]
        wordj=sent_token[i+1]
        if wordi not in forwardDict:
            wordi="UNK"
        if wordj not in forwardDict:
            wordj="UNK"
        if i==0:
            print(probability)
            print(probUniMatrix[forwardDict[wordi]])
            probability= probability*(1/probUniMatrix[forwardDict[wordi]])**(1/len(sent_token))
        else:
            print(probability)
            print(probMtrx[forwardDict[wordi]][forwardDict[wordj]])
            probMtrx[forwardDict[wordi]][forwardDict[wordj]]
            probability=probability*(1/probMtrx[forwardDict[wordi]][forwardDict[wordj]])**(1/len(sent_token))
    return(probability)
def perplexity(sentence,forwardDict,probMatrix,probUniMatrix):
    sent_token=word_tokenize(sentence)
    return(1/(sentence_probability(sent_token,probMatrix, forwardDict,probUniMatrix))**(1/len(sent_token)))
    

def randrange_float(start, stop, step):
    return random.randint(0, int((stop - start) / step)) * step + start

def generate1(forwardDict,backwardsDict,probMatrix,probUniMatrix,totalProb):
    sentence=[]
    firstWordProb=0
    for i in probUniMatrix:
        firstWordProb+=i
    firstWordProb=firstWordProb-probUniMatrix[len(probUniMatrix)-1]
    numberUpTo=randrange_float(0,firstWordProb,probUniMatrix[len(probUniMatrix)-1])
    print(numberUpTo)
    j=0
    while numberUpTo>0 and j<len(probUniMatrix)-2:
        numberUpTo=numberUpTo-probUniMatrix[j]
        print(probUniMatrix[j])
        print(numberUpTo)
        j=j+1
    print(j)
    print(backwardsDict[j])
    sentence.append(backwardsDict[j])
    numberUpTo=0
    nextWordProb=0
    word=backwardsDict[j]
    while word!="." and word!="!" and word!="?":
        nextWordProb=0
        for k in range(len(probMatrix[j])-1):
            nextWordProb+=probMatrix[j][k]
        print(numberUpTo)
        nextWordProb=nextWordProb-probMatrix[j][len(probMatrix[j])-1]
        numberUpTo=randrange_float(0,nextWordProb,probMatrix[j][len(probMatrix)-1])
        l=0
        while numberUpTo>0 and l<len(probMatrix[j])-1:
            numberUpTo=numberUpTo-probMatrix[j][l]
            print(probMatrix[j][l])
            print(numberUpTo)
            l=l+1
        print(l)
        print(backwardsDict[l])
        j=l
        sentence.append(backwardsDict[l])
        word=backwardsDict[l]
        print(sentence)
        print(backwardsDict[j])
    

    
def generate(startword,forwardDict,backwardsDict,probMatrix,probUniMatrix,totalProb):
    if startword not in forwardDict:
        print("bad")
        generate1(forwardDict,backwardsDict,probMatrix,probUniMatrix,totalProb)
    else:
        sentence=[]
        sentence.append(startword)
        firstWordProb=0
        j=forwardDict[startword]
        numberUpTo=0
        nextWordProb=0
        word=backwardsDict[j]
        while word!="." and word!="!" and word!="?":
            nextWordProb=0
            print("j j",j)
            for k in range(len(probMatrix[j])-1):
                nextWordProb+=probMatrix[j][k]
            print(numberUpTo)
            nextWordProb=nextWordProb-probMatrix[j][len(probMatrix[j])-1]
            numberUpTo=randrange_float(0,nextWordProb,probMatrix[j][len(probMatrix)-1])
            l=0
            while numberUpTo>0 and l<len(probMatrix[j])-1:
                numberUpTo=numberUpTo-probMatrix[j][l]
                print(probMatrix[j][l])
                print(numberUpTo)
                l=l+1
            print(l)
            print(backwardsDict[l])
            j=l
            sentence.append(backwardsDict[l])
            word=backwardsDict[l]
            print(sentence)
            print(backwardsDict[j])
            print("j j" ,j)
        
    
    
    
def main():
    ##nltk.download('reuters')
    nltk.download('inaugural')
    nltk.download('punkt')
    docinaug=inaugural.fileids()
    documents = reuters.fileids()
    print(str(len(documents)))
    print(reuters.raw("test/15556"))
    forwardDict,backwardsDict,probMatrix,probUniMatrix,totalProb=tokenize(reuters.raw("test/15556"))

    ##print(documents[1])
    ##print(docinaug[1])
    #forwardDict,backwardDict,probMtrx=tokenize("the man. the man. the man")
    sent_token=word_tokenize("hello my friend how are you")
    print("a")
    print(sentence_perplex(inaugural.raw(docinaug[1]),probMatrix,forwardDict,probUniMatrix))
    ##perplex_of_corpus=perplexity(inaugural.raw(docinaug[1]),forwardDict,probMatrix,probUniMatrix) 
    #generate("the",forwardDict,backwardsDict,probMatrix,probUniMatrix,totalProb)
main()
        
