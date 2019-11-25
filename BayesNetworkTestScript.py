#!/usr/bin/env python3

from BayesianNetworks import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#############################
## Example Tests from Bishop Pattern recognition textbook on page 377
#############################
BatteryState = readFactorTable(['battery'], [0.9, 0.1], [[1, 0]])
FuelState = readFactorTable(['fuel'], [0.9, 0.1], [[1, 0]])
GaugeBF = readFactorTable(['gauge', 'battery', 'fuel'], [0.8, 0.2, 0.2, 0.1, 0.2, 0.8, 0.8, 0.9], [[1, 0], [1, 0], [1, 0]])

carNet = [BatteryState, FuelState, GaugeBF] # carNet is a list of factors 
## Notice that different order of operations give the same answer
## (rows/columns may be permuted)
# joinFactors(joinFactors(BatteryState, FuelState), GaugeBF)
#joinFactors(joinFactors(GaugeBF, FuelState), BatteryState)

#marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'gauge')
#joinFactors(marginalizeFactor(GaugeBF, 'gauge'), BatteryState)

#joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState)
#marginalizeFactor(joinFactors(joinFactors(GaugeBF, FuelState), BatteryState), 'battery')

#marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'gauge')
#marginalizeFactor(joinFactors(marginalizeFactor(joinFactors(GaugeBF, BatteryState), 'battery'), FuelState), 'fuel')

#evidenceUpdateNet(carNet, ['fuel'], [1])
#evidenceUpdateNet(carNet, ['fuel', 'battery'], [1, 0])

## Marginalize must first combine all factors involving the variable to
## marginalize. Again, this operation may lead to factors that aren't
## probabilities.
#marginalizeNetworkVariables(carNet, ['battery']) ## this returns back a list
#marginalizeNetworkVariables(carNet, ['fuel']) ## this returns back a list
#marginalizeNetworkVariables(carNet, ['battery', 'fuel'])

# inference
print("inference starts")

print(inference(carNet, ['battery', 'fuel'], [], []) )        ## chapter 8 equation (8.30)
print(inference(carNet, ['battery'], ['fuel'], [0]))           ## chapter 8 equation (8.31)
print(inference(carNet, ['battery'], ['gauge'], [0]))          ##chapter 8 equation  (8.32)
print(inference(carNet, [], ['gauge', 'battery'], [0, 0]))    ## chapter 8 equation (8.33)
print("inference ends")
###########################################################################
#RiskFactor Data Tests
###########################################################################
riskFactorNet = pd.read_csv('RiskFactorsData.csv')

# Create factors

income      = readFactorTablefromData(riskFactorNet, ['income'])
smoke       = readFactorTablefromData(riskFactorNet, ['smoke', 'income'])

cholesterol = readFactorTablefromData(riskFactorNet,['cholesterol','exercise','smoke','income'])
bp          = readFactorTablefromData(riskFactorNet,['bp','exercise','smoke','income'])
exercise    = readFactorTablefromData(riskFactorNet, ['exercise', 'income'])
bmi         = readFactorTablefromData(riskFactorNet, ['bmi', 'income'])

diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi'])
angina      = readFactorTablefromData(riskFactorNet,['angina','cholesterol','bmi','bp'])
attack      = readFactorTablefromData(riskFactorNet,['attack','cholesterol','bmi','bp'])
stroke      = readFactorTablefromData(riskFactorNet,['stroke','cholesterol','bmi','bp'])

## you need to create more factor tables

risk_net_diabetes = [income, smoke, exercise, bmi, diabetes]
risk_net_stroke = [stroke,bmi,bp,cholesterol,income,exercise,smoke]
risk_net_attack = [attack,bmi,bp,cholesterol,income,exercise,smoke]
risk_net_angina = [angina,bmi,bp,cholesterol,income,exercise,smoke]

risk_net = [risk_net_angina,risk_net_attack,risk_net_diabetes,risk_net_stroke]


factors = riskFactorNet.columns

# example test p(diabetes|smoke=1,exercise=2)



### Please write your own test scrip similar to  the previous example 
###########################################################################
#HW4 test scrripts start from here
###########################################################################

#Exercise 2a
#a)
print("----------------------------------------------------------------------")
print("EXERCISE 2A")
def queryEx2a(risk_net):
    
    margVars_attack = list(set(factors)-{'attack', 'smoke', 'exercise'})
    margVars_stroke = list(set(factors)-{'stroke','smoke','exercise'})
    margVars_angina = list(set(factors)-{'angina','smoke','exercise'})
    margVars_diabetes = list(set(factors)-{'diabetes','smoke','exercise'})

    margVars = [margVars_angina,margVars_attack,margVars_diabetes,margVars_stroke]
    obsVars  = ['smoke', 'exercise']
    obsVals_bad  = [1, 2]
    obsVals_good = [2, 1]

    for i in range(len(margVars)):
        p_bad = inference(risk_net[i], margVars[i],obsVars, obsVals_bad)
        p_good = inference(risk_net[i],margVars[i],obsVars,obsVals_good)
        print("Probability table bad habits:")
        print(p_bad)
        print("Probability good habits:")
        print(p_good)

queryEx2a(risk_net)
#b)
print("----------------------------------------------------------------------")
print("EXERCISE 2B")
def queryEx2b(risk_net):
    
    margVars_attack = list(set(factors)-{'attack', 'bp', 'cholesterol','bmi'})
    margVars_stroke = list(set(factors)-{'stroke','bp','cholesterol','bmi'})
    margVars_angina = list(set(factors)-{'angina','bp','cholesterol','bmi'})
    margVars_diabetes = list(set(factors)-{'diabetes','bp','cholesterol','bmi'})
    margVars = [margVars_angina,margVars_attack,margVars_diabetes,margVars_stroke]
    obsVars = ['bp','cholesterol','bmi']
    obsVals_bad = [1,1,3]
    obsVals_good = [2,2,2]
    for i in range(len(margVars)):
        p_bad = inference(risk_net[i],margVars[i],obsVars,obsVals_bad)
        p_good = inference(risk_net[i],margVars[i],obsVars,obsVals_good)
        print("Probability table bad health")
        print(p_good)
        print("Probability table good health")
        print(p_bad)
queryEx2b(risk_net)

#Exercise 3
print("----------------------------------------------------------------------")
print("EXERCISE 3")
def ex3():

    margVars_angina = list(set(factors)-{'income','angina'})
    margVars_attack = list(set(factors)-{'income','attack'})
    margVars_diabetes = list(set(factors)-{'income','diabetes'})
    margVars_stroke = list(set(factors)-{'income','stroke'})


    margVars = [margVars_angina,margVars_attack,margVars_diabetes,margVars_stroke]

    obsVarsList = [['angina'],['attack'],['diabetes'],['stroke']]
    obsVals = [1]


    for i in range(len(margVars)):
        p = inference(risk_net[i],margVars[i],[],[])
        p = evidenceUpdateNet([p],obsVarsList[i],obsVals)[0]
        print(p)
        print(obsVarsList[i][0])
        p.plot(kind='bar', x='income',y='probs')
        desease = ['Angina','a heart attack','diabetes','a stroke']
        plt.title("Probability of having "+desease[i]+" at different income levels")
        plt.xlabel = "Income"
        plt.ylabel = "P("+obsVarsList[i][0]+"|income)"
        plt.show()
ex3()
#4

def ex4():
    print("----------------------------------------------------------------------")
    print("EXERCISE 4A")
    diabetes    = readFactorTablefromData(riskFactorNet, ['diabetes', 'bmi','smoke','exercise'])
    angina      = readFactorTablefromData(riskFactorNet,['angina','cholesterol','bmi','bp','smoke','exercise'])
    attack      = readFactorTablefromData(riskFactorNet,['attack','cholesterol','bmi','bp','smoke','exercise'])
    stroke      = readFactorTablefromData(riskFactorNet,['stroke','cholesterol','bmi','bp','smoke','exercise'])

    risk_net_diabetes = [income, smoke, exercise, bmi, diabetes]
    risk_net_stroke = [stroke,bmi,bp,cholesterol,income,exercise,smoke]
    risk_net_attack = [attack,bmi,bp,cholesterol,income,exercise,smoke]
    risk_net_angina = [angina,bmi,bp,cholesterol,income,exercise,smoke]

    risk_net2 = [risk_net_angina,risk_net_attack,risk_net_diabetes,risk_net_stroke]

    queryEx2a(risk_net2)
    print("----------------------------------------------------------------------")
    print("EXERCISE 4B")
    queryEx2b(risk_net)
#queryEx2a(risk_net)

def ex5():
    print("----------------------------------------------------------------------")
    print("EXERCISE 5")
    stroke=readFactorTablefromData(riskFactorNet,['stroke','cholesterol','bmi','bp','smoke','exercise','diabetes'])
    risk_net_stroke = [stroke,bmi,bp,cholesterol,income,exercise,smoke]
    #risk_net3 = [risk_net_angina,risk_net_attack,risk_net_diabetes,risk_net_stroke]

    margVars = list(set(factors)-{'stroke','diabetes'})
    obsVars = []
    obsVals = []
    p = inference(risk_net_stroke,margVars,obsVars,obsVals)
    p1 = evidenceUpdateNet([p],['stroke','diabetes'],[1,1])[0]
    print(p1)
    p2 = evidenceUpdateNet([p],['stroke','diabetes'],[1,3])[0]
    print(p2)

ex5()







