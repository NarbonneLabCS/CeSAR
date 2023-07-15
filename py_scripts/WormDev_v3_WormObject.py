#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
from datetime import datetime
import math
import pprint
import matplotlib

print()
print("     dBBBPdBP dP dBBBBP          dBPdBPdBP dBBBBP dBBBBBb    dBBBBBBb")
print("                dBP.BP                    dBP.BP      dBP         dBP")
print("   dBBP dB .BP dBP.BP          dBPdBPdBP dBP.BP   dBBBBK   dBPdBPdBP ")
print("  dBP   BB.BP dBP.BP dBBBBBP  dBPdBPdBP dBP.BP   dBP  BB  dBPdBPdBP  ")
print(" dBBBBP BBBP dBBBBP          dBBBBBBBP dBBBBP   dBP  dB' dBPdBPdBP   ")
print()

def inputChecker(inputQuestion, regexMatch, errorMessage):
    temp = input(inputQuestion)
    while not re.match(regexMatch, temp):
        print(errorMessage)
        temp = input(inputQuestion)
    else:
        #print("Valid input.")
        return temp

inputQuestion = "How many do you estimate (1 to 500)?   -   "
regexUpperNbLimit = "^([]*|[0-9]|[1-8][0-9]|9[0-9]|[1-4][0-9]{2}|500)$"
upperNbLimitReachedError = "Error! Please make sure you input a number between 1 and 500."

def inputHowManyQuestionFormatter(stageSearched, inputQuestion, indexWord):
    return (inputQuestion[:inputQuestion.find(indexWord)] + stageSearched + inputQuestion[inputQuestion.find(indexWord):])

initEggNb = inputChecker(inputHowManyQuestionFormatter("eggs ", inputQuestion, "do"), regexUpperNbLimit, upperNbLimitReachedError)

initL1Nb = inputChecker(inputHowManyQuestionFormatter("L1 ", inputQuestion, "do"), regexUpperNbLimit, upperNbLimitReachedError)

initL2Nb = inputChecker(inputHowManyQuestionFormatter("L2 ", inputQuestion, "do"), regexUpperNbLimit, upperNbLimitReachedError)

initL3Nb = inputChecker(inputHowManyQuestionFormatter("L3 ", inputQuestion, "do"), regexUpperNbLimit, upperNbLimitReachedError)

initL4Nb = inputChecker(inputHowManyQuestionFormatter("L4 ", inputQuestion, "do"), regexUpperNbLimit, upperNbLimitReachedError)

initAdultsNb = inputChecker(inputHowManyQuestionFormatter("adults ", inputQuestion, "do"), regexUpperNbLimit, upperNbLimitReachedError)

initEggLayingNb = inputChecker(inputHowManyQuestionFormatter("egg-laying ", inputQuestion, "do"), regexUpperNbLimit, upperNbLimitReachedError)

wormPop = [initEggNb, initL1Nb, initL2Nb, initL3Nb, initL4Nb, initAdultsNb, initEggLayingNb]

# datetime object containing current date and time
now = datetime.now()
 
print("Current datetime: ", now)

# dd/mm/YY H:M:S
datetime = now.strftime("%d/%m/%Y %H:%M:%S")

print("date and time =", datetime)
curMin= now.strftime("%M")
curHour = now.strftime("%H")
print("curHour: " + curHour)
print("curMin: " + curMin)
        
stageTTDict = {
  "116": 17,
  "120": 11,
  "125": 9,

  "216": 16,
  "220": 9,
  "225": 8,

  "316": 12,
  "320": 9,
  "325": 6,

  "416": 15,
  "420": 13,
  "425": 8,

  "516": 14,
  "520": 10,
  "525": 6,

  "616": 15,
  "620": 9,
  "625": 8,

  "716": 90,
  "720": 63,
  "725": 41,
}

nbDays = 5
nbHours = 24
nbWormStages = 7
nbFilialis = 3
nbFilVar = 2 # one for stage and another for time remaining

workweek16C = []
workweek20C = []
workweek25C = []

def CreateWorkweek (week):
    for day in range(nbDays):
        week.append([])

        for hour in range(nbHours):
            week[day].append([])

            for fil in range(nbFilialis):
                week[day][hour].append([])

                for stage in range(nbWormStages):
                    week[day][hour][fil].append([])

                    #for var in range(nbFilVar):
                    #    week[day][hour][fil][stage].append(0)

def SetFirstHour(week, temperature):
    hour = int(curHour)
    for i in range(int(initEggNb)):
        week[0][hour][0][0].append(stageTTDict["1" + temperature])
    for i in range(int(initL1Nb)):
        week[0][hour][0][1].append(stageTTDict["2" + temperature])
    for i in range(int(initL2Nb)):
        week[0][hour][0][2].append(stageTTDict["3" + temperature])
    for i in range(int(initL3Nb)):
        week[0][hour][0][3].append(stageTTDict["4" + temperature])
    for i in range(int(initL4Nb)):
        week[0][hour][0][4].append(stageTTDict["5" + temperature])
    for i in range(int(initAdultsNb)):
        week[0][hour][0][5].append(stageTTDict["6" + temperature])
    for i in range(int(initEggLayingNb)):
        week[0][hour][0][6].append(stageTTDict["7" + temperature])

def EggLayingRateFunc(fTemp, hour):
  y = 0
  x = int(hour)
  #change proportion 0 to 1
  if(fTemp == 16):#lagrange interpolation and gaussian function
    y = round((5.4 * 3 * math.exp(-(x - 45)**2 / (2 * 9.377701**2)) + (- 0.002666666666666 * x**2 + 0.24 * x) * 10) / 13)
  if(fTemp == 20):
    y = round((8.1 * 12 * math.exp(-(x - 31.5)**2 / (2 * 6.279314**2)) + (- 0.00816327 * x**2 + 0.514286 * x) * 31) / 43)
  if(fTemp == 25):
    y = round((9.1 * 16 * math.exp(-(x - 20.5)**2 / (2 * 4.086538**2)) + (-0.0216538 * x**2 + 0.887805 * x) * 15) / 31)
  return y

def Evolve(week, evoTemp):
    maxHour = nbDays * nbHours
    #print("maxHour: " + str(maxHour))
    remainHour = maxHour - int(curHour)
    #print("remainHour: " + str(remainHour))
    d = 0
    h = int(curHour)
    

    while remainHour > 1:

        broodingRate = 0
        #print("[" + str(d+1) + "/" + str(h) + "]")

        # goto next day
        if(h == 23): 
            d = d + 1
            h = 0

            # week[day][hour][f][stg].append(0)
            for f in range(nbFilialis):   
                for stg in range(6, -1, -1):
                    for i in range(len(week[d-1][23][f][stg]), 0, -1): 
                        
                        if(week[d-1][23][f][stg][i-1] == 1):
                            if(stg == 6 and f < 2):
                                week[d][h][f+1][0].append(stageTTDict["1" + evoTemp])
                            if(stg < 6):
                                week[d][h][f][stg+1].append(stageTTDict[str(stg+2) + evoTemp]) 
                        elif(week[d-1][23][f][stg][i-1] > 1):
                            if(stg == 6 and f < 2): 
                                broodingRateX = stageTTDict["7" + evoTemp] - week[d-1][23][f][stg][i-1]
                                broodingRate = EggLayingRateFunc(int(evoTemp), broodingRateX)
                                for j in range(broodingRate):
                                  week[d][h][f+1][0].append(stageTTDict["1" + evoTemp])
                                week[d][h][f][stg].append(week[d-1][23][f][stg][i-1]-1)
                            elif(stg < 6):
                                week[d][h][f][stg].append(week[d-1][23][f][stg][i-1]-1)

        # goto next hour        
        else: 
            h = h + 1

            for f in range(nbFilialis):
                for stg in range(6, -1, -1):
                    for i in range(len(week[d][h-1][f][stg]), 0, -1):
                        
                        if(week[d][h-1][f][stg][i-1] == 1):
                            if(stg == 6 and f < 2):
                                week[d][h][f+1][0].append(stageTTDict["1" + evoTemp])
                            elif(stg < 6):
                                week[d][h][f][stg+1].append(stageTTDict[str(stg+2) + evoTemp]) 
                            
                        elif(week[d][h-1][f][stg][i-1] > 1):
                            if(stg == 6 and f < 2):
                                broodingRateX = stageTTDict["7" + evoTemp] - week[d][h-1][f][stg][i-1]
                                broodingRate = EggLayingRateFunc(int(evoTemp), broodingRateX)
                                for j in range(broodingRate):
                                  week[d][h][f+1][0].append(stageTTDict["1" + evoTemp])
                                week[d][h][f][stg].append(week[d][h-1][f][stg][i-1]-1)

                            elif(stg < 6):
                                week[d][h][f][stg].append(week[d][h-1][f][stg][i-1]-1)

        remainHour = remainHour - 1


CreateWorkweek(workweek16C)
SetFirstHour(workweek16C, "16")
Evolve(workweek16C, "16")

CreateWorkweek(workweek20C)
SetFirstHour(workweek20C, "20")
Evolve(workweek20C, "20")

CreateWorkweek(workweek25C)
SetFirstHour(workweek25C, "25")
Evolve(workweek25C, "25")

pprint.pprint(workweek25C)

maxHour = nbDays * nbHours
remainHour = maxHour - int(curHour)


# libraries and data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
 


# --- FORMAT 2</pre>
x = range(0, 24)
stg0w = []
stg1w = []
stg2w = []
stg3w = []
stg4w = []
stg5w = []
stg6w = []



for d in range(0, 5):
    for i in range(0, 24, 1):
        globals()["stg0w"].append(len(workweek16C[d][i][0][0])+len(workweek16C[d][i][1][0])+len(workweek16C[d][i][2][0]))
        globals()["stg1w"].append(len(workweek16C[d][i][0][1])+len(workweek16C[d][i][1][1])+len(workweek16C[d][i][2][1]))
        globals()["stg2w"].append(len(workweek16C[d][i][0][2])+len(workweek16C[d][i][1][2])+len(workweek16C[d][i][2][2]))
        globals()["stg3w"].append(len(workweek16C[d][i][0][3])+len(workweek16C[d][i][1][3])+len(workweek16C[d][i][2][3]))
        globals()["stg4w"].append(len(workweek16C[d][i][0][4])+len(workweek16C[d][i][1][4])+len(workweek16C[d][i][2][4]))
        globals()["stg5w"].append(len(workweek16C[d][i][0][5])+len(workweek16C[d][i][1][5])+len(workweek16C[d][i][2][5]))
        globals()["stg6w"].append(len(workweek16C[d][i][0][6])+len(workweek16C[d][i][1][6])+len(workweek16C[d][i][2][6]))

from operator import add

vectStg0w = np.array(stg0w)
vectStg1w = np.array(stg1w)
vectStg2w = np.array(stg2w)
vectStg3w = np.array(stg3w)
vectStg4w = np.array(stg4w)
vectStg5w = np.array(stg5w)
vectStg6w = np.array(stg6w)

vectStgTotal = vectStg0w + vectStg1w + vectStg2w + vectStg3w + vectStg4w + vectStg5w + vectStg6w
#stgTotal = list(map(add, stg0w, stg1w, stg2w, stg3w, stg4w, stg5w, stg6w))

df=pd.DataFrame({'x': range(maxHour), 'egg': stg0w, 'L1': stg1w, 'L2': stg2w, 'L3': stg3w, 'L4': stg4w, 'adults': stg5w, 'egglaying': stg6w, 'total': vectStgTotal})
 
# style
plt.style.use('seaborn-darkgrid')
 
# create a color palette
palette = plt.get_cmap('Set1')
 
# multiple line plot
num=0
for column in df.drop('x', axis=1):
    num+=1
    plt.plot(df['x'], df[column], marker='', color=palette(num), linewidth=1, alpha=0.9, label=column)
 
# Add legend
plt.legend(loc=2, ncol=2)
 
# Add titles
plt.title("Evo-Worm", loc='left', fontsize=12, fontweight=0, color='orange')
plt.xlabel("Time (hour)")
plt.ylabel("Number")
plt.show()

import pandas as pd

graphLineNameDict = {
  "0": "egg",
  "1": "L1",
  "2": "L2",
  "3": "L3",
  "4": "L4",
  "5": "adult",
  "6": "eggLaying",
}

X = []
Y = []
Z = [] 
color = []

hour = 0
for d in range(0, 5):
    for h in range(0, 24, 1): # d and h allow access to values of every hour up to max hour
        for s in range (0, 7): # now looking at every stg
            for f in range (0, 3):
                tempTotal = 0
                x_i = 0
                for t in [16, 20, 25]:
                  X.append(s-0.2 + x_i)
                  Y.append(hour)
                  Z.append(len(globals()["workweek" + str(t) + "C"][d][i][f][s]))
                  #tempTotal += len(globals()["workweek" + str(t) + "C"][d][i][f][s])
                  #color.append(graphLineNameDict[str(s)] + "F" + str(f) + "T" + str(t))
                  
                  #Z.append(t)
                  #tempTotal += len(globals()["workweek" + str(t) + "C"][d][i][f][s])
                  color.append(graphLineNameDict[str(s)] + "F" + str(f) + "T" + str(t))
                  x_i += 0.2

        hour += 1

                #X.append(s-0.25 + f*0.25)
                #Y.append(d * h)
                #Z.append(tempTotal)            
                #color.append(graphLineNameDict[str(s)] + "F" + str(f) + "T" + str(t))

