import numpy as np
import pandas as pd
from functools import reduce

## Function to create a conditional probability table
## Conditional probability is of the form p(x1 | x2, ..., xk)
## varnames: vector of variable names (strings) first variable listed 
##           will be x_i, remainder will be parents of x_i, p1, ..., pk
## probs: vector of probabilities for the flattened probability table
## outcomesList: a list containing a vector of outcomes for each variable
## factorTable is in the type of pandas dataframe
## See the test file for examples of how this function works
def readFactorTable(varnames, probs, outcomesList):
    factorTable = pd.DataFrame({'probs': probs})

    totalfactorTableLength = len(probs)
    numVars = len(varnames)

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(totalfactorTableLength / (k * numLevs))
        k = k * numLevs
    return factorTable

## Build a factorTable from a data frame using frequencies
## from a data frame of data to generate the probabilities.
## data: data frame read using pandas read_csv
## varnames: specify what variables you want to read from the table
## factorTable is in the type of pandas dataframe
def readFactorTablefromData(data, varnames):
    numVars = len(varnames)
    outcomesList = []

    for i in range(0, numVars):
        name = varnames[i]
        outcomesList = outcomesList + [list(set(data[name]))]

    lengths = list(map(lambda x: len(x), outcomesList))
    m = reduce(lambda x, y: x * y, lengths)
    factorTable = pd.DataFrame({'probs': np.zeros(m)})

    k = 1
    for i in range(numVars - 1, -1, -1):
        levs = outcomesList[i]
        numLevs = len(levs)
        col = []
        for j in range(0, numLevs):
            col = col + [levs[j]] * k
        factorTable[varnames[i]] = col * int(m / (k * numLevs))
        k = k * numLevs

    numLevels = len(outcomesList[0])

    # creates the vector called fact to index probabilities 
    # using matrix multiplication with the data frame
    fact = np.zeros(data.shape[1])
    lastfact = 1
    for i in range(len(varnames) - 1, -1, -1):
        fact = np.where(np.isin(list(data), varnames[i]), lastfact, fact)
        lastfact = lastfact * len(outcomesList[i])

    # Compute unnormalized counts of subjects that satisfy all conditions
    a = (data - 1).dot(fact) + 1
    for i in range(0, m):
        factorTable.at[i,'probs'] = sum(a == (i+1))

    # normalize the conditional probabilities
    skip = int(m / numLevels)
    for i in range(0, skip):
        normalizeZ = 0
        for j in range(i, m, skip):
            normalizeZ = normalizeZ + factorTable['probs'][j]
        for j in range(i, m, skip):
            if normalizeZ != 0:
                factorTable.at[j,'probs'] = factorTable['probs'][j] / normalizeZ

    return factorTable


## Join of two factors
## factor1, factor2: two factor tables
##
## Should return a factor table that is the join of factor 1 and 2.
## You can assume that the join of two factors is a valid operation.
## Hint: You can look up pd.merge for mergin two factors
def joinFactors(factor1, factor2):
    #check if only common column is 'probs'
    if len(np.intersect1d(factor1.columns,factor2.columns))==1:
        #Cartesian product
        joined = pd.DataFrame()
        count = 0
        for i in factor1.index:
            for j in factor2.index:
                #calculating probability
                joined.loc[count,'probs'] = factor1.loc[i,'probs']*factor2.loc[j,'probs']
                f1column = factor1.columns.tolist()
                f2column = factor2.columns.tolist()
                f1column.remove('probs')
                f2column.remove('probs')
                for col in f1column:
                    joined.loc[count,col] = factor1.loc[i,col]
                for col in f2column:
                    joined.loc[count,col] = factor2.loc[j,col]
                count+=1
        return joined
        
    
    common_elements = np.intersect1d(factor1.columns,factor2.columns).tolist()
    common_elements.remove('probs')
        
    joined = pd.merge(factor1,factor2, how='inner', on=common_elements)
    joined['probs'] = joined['probs_x'].mul(joined['probs_y'])
    joined =joined.drop(columns=['probs_x','probs_y'])
    return joined
## Marginalize a variable from a factor
## table: a factor table in dataframe
## hiddenVar: a string of the hidden variable name to be marginalized
##
## Should return a factor table that marginalizes margVar out of it.
## Assume that hiddenVar is on the left side of the conditional.
## Hint: you can look can pd.groupby
def marginalizeFactor(factorTable, hiddenVar):
    var = factorTable.columns.tolist()
    var.remove(hiddenVar)
    var.remove('probs')
    group = factorTable.groupby(var)

    new_table = group['probs'].agg(np.sum).reset_index()

    return new_table

## Marginalize a list of variables 
## bayesnet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
##
## Should return a Bayesian network containing a list of factor tables that results
## when the list of variables in hiddenVar is marginalized out of bayesnet.
def marginalizeNetworkVariables(bayesNet, hiddenVar):
    for hVar in hiddenVar:
        joining_tables = []
        delete_table = []
        for i in range(len(bayesNet)):
        #Finds all tables that has the hidden variable in it
            if hVar in bayesNet[i].columns:
                joining_tables.append(bayesNet[i])
                delete_table.append(i)
        if len(joining_tables)>0:    
            #Joins all the tables together until it consist of one dataframe
            while len(joining_tables)>1:
                joining_tables[0] = joinFactors(joining_tables[0],joining_tables[1])
                del joining_tables[1]
            
            temp_bayesNet = []
            for i in range(len(bayesNet)):
                if i not in delete_table:
                    temp_bayesNet.append(bayesNet[i])
            bayesNet = temp_bayesNet
            joining_tables[0] = marginalizeFactor(joining_tables[0],hVar)
            bayesNet.append(joining_tables[0])
        
        #print(bayesNet)         
    return bayesNet

## Update BayesNet for a set of evidence variables
## bayesNet: a list of factor and factor tables in dataframe format
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## Set the values of the evidence variables. Other values for the variables
## should be removed from the tables. You do not need to normalize the factors
def evidenceUpdateNet(bayesNet, evidenceVars, evidenceVals):
    for i in range(len(bayesNet)):
        for j in range(len(evidenceVars)):
            if evidenceVars[j] in bayesNet[i].columns:
                #print(evidenceVars,evidenceVals)
                bayesNet[i] = bayesNet[i][bayesNet[i][evidenceVars[j]]==evidenceVals[j]]
                #bayesNet[i] = bayesNet

    return bayesNet


## Run inference on a Bayesian network
## bayesNet: a list of factor tables and each table iin dataframe type
## hiddenVar: a string of the variable name to be marginalized
## evidenceVars: a vector of variable names in the evidence list
## evidenceVals: a vector of values for corresponding variables (in the same order)
##
## This function should run variable elimination algorithm by using 
## join and marginalization of the sets of variables. 
## The order of the elimiation can follow hiddenVar ordering
## It should return a single joint probability table. The
## variables that are hidden should not appear in the table. The variables
## that are evidence variable should appear in the table, but only with the single
## evidence value. The variables that are not marginalized or evidence should
## appear in the table with all of their possible values. The probabilities
## should be normalized to sum to one.
def inference(bayesNet, hiddenVar, evidenceVars, evidenceVals):
    bayesNet = bayesNet.copy()
    bayesNet = evidenceUpdateNet(bayesNet,evidenceVars,evidenceVals)
    bayesNet = marginalizeNetworkVariables(bayesNet,hiddenVar)

    while len(bayesNet)>1:
        bayesNet[0] = joinFactors(bayesNet[0],bayesNet[1])
        del bayesNet[1]

    probTable = bayesNet[0]
    s = probTable['probs'].sum()
    probTable['probs'] = probTable['probs'].divide(s)
    return probTable





