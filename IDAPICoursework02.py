#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Coursework in Python 
from IDAPICourseworkLibrary import *
from numpy import *
#
# Coursework 1 begins here
#
# Function to compute the prior distribution of the variable root from the data set
def Prior(theData, root, noStates):
    prior = zeros((noStates[root]), float )
    for i in range (0,noStates[root]):
        prior[i] = float(list(theData[:,root]).count(i))/len(list(theData[:,0]))
    
# end of Coursework 1 task 1
    return prior
# Function to compute a CPT with parent node varP and xchild node varC from the data array
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT(theData, varC, varP, noStates):
    cPT = zeros((noStates[varC], noStates[varP]), float )
    for i in range (0,len(list(theData[:,0]))):
        parentDataPoint = theData[i][varP];
        childDataPoint = theData[i][varC];
        cPT[childDataPoint][parentDataPoint] = cPT[childDataPoint][parentDataPoint] + 1
    for j in range (0,noStates[varP]):
        cPT[:,j] = cPT[:,j] / float(list(theData[:,varP]).count(j))
# end of coursework 1 task 2
    return cPT
# Function to calculate the joint probability table of two variables in the data set
def JPT(theData, varRow, varCol, noStates):
    jPT = zeros((noStates[varRow], noStates[varCol]), float )
#Coursework 1 task 3 should be inserted here
    for i in range (0,len(list(theData[:,0]))):
        rowDataPoint = theData[i][varRow];
        colDataPoint = theData[i][varCol];
        jPT[rowDataPoint][colDataPoint] = jPT[rowDataPoint][colDataPoint] + 1
    jPT = jPT / len(list(theData[:,0]))     
# end of coursework 1 task 3
    return jPT
#
# Function to convert a joint probability table to a conditional probability table
def JPT2CPT(aJPT):
#Coursework 1 task 4 should be inserted here 
    noRows = aJPT.shape[0]
    noCols = aJPT.shape[1]
    for j in range(0,noCols):
        marginal = sum(aJPT[:,j])
        aJPT[:,j] = aJPT[:,j] / marginal    
# coursework 1 taks 4 ends here
    return aJPT

#
# Function to query a naive Bayesian network
def Query(theQuery, naiveBayes): 
    rootPdf = zeros((naiveBayes[0].shape[0]), float)
# Coursework 1 task 5 should be inserted here
    for i in range(0,naiveBayes[0].shape[0]):
        rootPdf[i] = naiveBayes[0][i]
        for j in range(0,len(theQuery)):
            rootPdf[i] =  rootPdf[i] * naiveBayes[j+1][theQuery[j]][i]
    if sum(rootPdf) != 0:
       rootPdf = rootPdf / sum(rootPdf)
# end of coursework 1 task 5
    return rootPdf
#
# End of Coursework 1
#
# Coursework 2 begins here
#
# Calculate the mutual information from the joint probability table of two variables
def MutualInformation(jP):
    mi=0.0
# Coursework 2 task 1 should be inserted here
    # calculate marginal distribution of variables
    marginalA = numpy.sum(jP,axis = 1)
    marginalB = numpy.sum(jP,axis = 0)
    # calculate mutual information using formula from lecture 6
    for i in range(0,jP.shape[0]):
        for j in range(0,jP.shape[1]):
            if (jP[i][j] != 0) and (marginalA[i] != 0) and (marginalB[j] != 0):
                mi = mi + (jP[i][j] * math.log(jP[i][j]/(marginalA[i]*marginalB[j]),2))
# end of coursework 2 task 1
    return mi
#
# construct a dependency matrix for all the variables
def DependencyMatrix(theData, noVariables, noStates):
    MIMatrix = zeros((noVariables,noVariables))
# Coursework 2 task 2 should be inserted here
    # (i,j)-th entry of matrix is the mutual information between variables i and j
    for i in range(0,noVariables):
        for j in range(0,noVariables):
            MIMatrix[i][j] = MutualInformation(JPT(theData,i,j,noStates))
# end of coursework 2 task 2
    return MIMatrix
# Function to compute an ordered list of dependencies 
def DependencyList(depMatrix):
    depList=[]
# Coursework 2 task 3 should be inserted here
    # only consider n(n-1)/2 edges in dependency list as we exclude self-edges
    for i in range(0,depMatrix.shape[1]):
        for j in range(i+1,depMatrix.shape[1]): 
            depList.append([depMatrix[i][j],i,j])
    # sort the dependency list on the dependency values
    depList.sort(key=lambda x: float(x[0]), reverse = True)
# end of coursework 2 task 3
    return array(depList)
#
# Functions implementing the spanning tree algorithm
# Coursework 2 task 4

def SpanningTreeAlgorithm(depList, noVariables):
    spanningTree = []
    # the (i,j)-th entry of the accessibility matrix = 1 if there exists a path between variables i and j in the network and = 0 otherwise.
    accessibilityMatrix = zeros((noVariables, noVariables), int)
    outputFile = "accessibilityMatrix.txt"#
    # iterate through all edges in dependency list in descending order of dependency
    for i in range(0,len(depList)):
        # extract nodes in edge, call them begin and end
        beginNode = depList[i][1]
        endNode = depList[i][2]
        # add edge to spanning tree if a path does not already exist between the nodes
        if accessibilityMatrix[beginNode][endNode] == 0:
            spanningTree.append(depList[i])
            # update accessibility matrix now we have added the edge
            accessibilityMatrix[beginNode][endNode] = 1
            accessibilityMatrix[endNode][beginNode] = 1
            # connect each node accessible from the end node (except from begin node) to the all nodes accessible from the begin node (including the begin node but except from the end node)
            for endNodeAccessible in range(0,noVariables):
                if ((accessibilityMatrix[endNodeAccessible][endNode] == 1) or (endNodeAccessible == endNode))and (endNodeAccessible != beginNode):
                    for beginNodeAccessible in range(0,noVariables):
                        if ((accessibilityMatrix[beginNodeAccessible][beginNode] == 1) or (beginNodeAccessible == beginNode)) and (beginNodeAccessible != endNode):
                            accessibilityMatrix[beginNodeAccessible][endNodeAccessible] = 1
                            accessibilityMatrix[endNodeAccessible][beginNodeAccessible] = 1   
    return array(spanningTree)
#
# End of coursework 2
#
# Coursework 3 begins here
#
# Function to compute a CPT with multiple parents from he data set
# it is assumed that the states are designated by consecutive integers starting with 0
def CPT_2(theData, child, parent1, parent2, noStates):
    cPT = zeros([noStates[child],noStates[parent1],noStates[parent2]], float )
# Coursework 3 task 1 should be inserted here
<<<<<<< HEAD
    
=======
   
>>>>>>> ba56d4cea02bd738d894a37e87abd6f46a885f48

# End of Coursework 3 task 1           
    return cPT
#
# Definition of a Bayesian Network
def ExampleBayesianNetwork(theData, noStates):
    arcList = [[0],[1],[2,0],[3,2,1],[4,3],[5,3]]
    cpt0 = Prior(theData, 0, noStates)
    cpt1 = Prior(theData, 1, noStates)
    cpt2 = CPT(theData, 2, 0, noStates)
    cpt3 = CPT_2(theData, 3, 2, 1, noStates)
    cpt4 = CPT(theData, 4, 3, noStates)
    cpt5 = CPT(theData, 5, 3, noStates)
    cptList = [cpt0, cpt1, cpt2, cpt3, cpt4, cpt5]
    return arcList, cptList
# Coursework 3 task 2 begins here

# end of coursework 3 task 2
#
# Function to calculate the MDL size of a Bayesian Network
def MDLSize(arcList, cptList, noDataPoints, noStates):
    mdlSize = 0.0
# Coursework 3 task 3 begins here


# Coursework 3 task 3 ends here 
    return mdlSize 
#
# Function to calculate the joint probability of a single data point in a Network
def JointProbability(dataPoint, arcList, cptList):
    jP = 1.0
# Coursework 3 task 4 begins here


# Coursework 3 task 4 ends here 
    return jP
#
# Function to calculate the MDLAccuracy from a data set
def MDLAccuracy(theData, arcList, cptList):
    mdlAccuracy=0
# Coursework 3 task 5 begins here


# Coursework 3 task 5 ends here 
    return mdlAccuracy
#
# End of coursework 3
#
# Coursework 4 begins here
#
def Mean(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    mean = []
    # Coursework 4 task 1 begins here



    # Coursework 4 task 1 ends here
    return array(mean)


def Covariance(theData):
    realData = theData.astype(float)
    noVariables=theData.shape[1] 
    covar = zeros((noVariables, noVariables), float)
    # Coursework 4 task 2 begins here


    # Coursework 4 task 2 ends here
    return covar
def CreateEigenfaceFiles(theBasis):
    adummystatement = 0 #delete this when you do the coursework
    # Coursework 4 task 3 begins here

    # Coursework 4 task 3 ends here

def ProjectFace(theBasis, theMean, theFaceImage):
    magnitudes = []
    # Coursework 4 task 4 begins here

    # Coursework 4 task 4 ends here
    return array(magnitudes)

def CreatePartialReconstructions(aBasis, aMean, componentMags):
    adummystatement = 0  #delete this when you do the coursework
    # Coursework 4 task 5 begins here

    # Coursework 4 task 5 ends here

def PrincipalComponents(theData):
    orthoPhi = []
    # Coursework 4 task 3 begins here
    # The first part is almost identical to the above Covariance function, but because the
    # data has so many variables you need to use the Kohonen Lowe method described in lecture 15
    # The output should be a list of the principal components normalised and sorted in descending 
    # order of their eignevalues magnitudes

    
    # Coursework 4 task 6 ends here
    return array(orthoPhi)

#
# main program part for Coursework 1
#
"""outputFile = "IDAPIResults01.txt"
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("Neurones.txt")
theData = array(datain)
#print theData
AppendString(outputFile,"Coursework One Results by Hesam Ipakchi (00648378), Yijie Ge (00650073)")
AppendString(outputFile,"") #blank line
AppendString(outputFile,"The prior probability of node 0")
prior = Prior(theData, 0, noStates)
AppendList(outputFile, prior)
AppendString(outputFile,"The conditional probability table P(2|0)")
CPt2 = CPT(theData, 2, 0 , noStates)
AppendArray(outputFile, CPt2)
JPt = JPT(theData, 2, 0 , noStates)
AppendString(outputFile,"The joint probability table between P(2&0)")
AppendArray(outputFile, JPt)
JPt2CPt = JPT2CPT(JPt)
AppendString(outputFile,"The conditional probability table P(2|0) calculated from joint probability table P(2&0)")
AppendArray(outputFile,JPt2CPt)
CPt1 = CPT(theData, 1, 0 , noStates)
CPt3 = CPT(theData, 3, 0 , noStates)
CPt4 = CPT(theData, 4, 0 , noStates)
CPt5 = CPT(theData, 5, 0 , noStates)
theQuery1 = [4,0,0,0,5]
theQuery2 = [6,5,2,5,5]
naiveBayes = [array(prior),CPt1,CPt2,CPt3,CPt4,CPt5]
queryResult1 = Query(theQuery1,naiveBayes)
queryResult2 = Query(theQuery2,naiveBayes)
AppendString(outputFile, "result of query " + str(theQuery1) + " :")
AppendList(outputFile, queryResult1)
AppendString(outputFile, "result of query " + str(theQuery2) + " :")
AppendList(outputFile, queryResult2)
"""
#
# main program part for Coursework 2
#

# Output results in text file before transferring to pdf
outputFile = "IDAPIResults02.txt"
<<<<<<< HEAD
AppendString(outputFile,"Coursework Two Results by Hesam Ipakchi (00648378), Yijie Ge (00650073), Joysen Goes (00649833)")
=======
AppendString(outputFile,"Coursework Two Results by Hesam Ipakchi (00648378), Yijie Ge (00650073)")
>>>>>>> ba56d4cea02bd738d894a37e87abd6f46a885f48
AppendString(outputFile,"") #blank line
AppendString(outputFile,"The Dependency matrix for HepatitisC data set")
noVariables, noRoots, noStates, noDataPoints, datain = ReadFile("HepatitisC.txt")
theData = array(datain)
MIMatrix = DependencyMatrix(theData, noVariables,noStates) 
AppendArray(outputFile,MIMatrix)
depList = DependencyList(MIMatrix)
AppendString(outputFile,"The Dependency list for HepatitisC data set")
AppendArray(outputFile,depList)
spanningTree = SpanningTreeAlgorithm(depList, noVariables)
AppendString(outputFile,"The spanning tree found for HepatitisC data set")
AppendArray(outputFile,spanningTree)
