from main2 import *
import matplotlib.pyplot as plt
import seaborn
import pandas as pd

print("Import data")
dataTrue, title = readcsv()
data = part(dataTrue)
print("Data imported")

data2 = []
title2 = []

for j in range(0, len(data)):
    data2.append(data[j][0:14])
    data2[j].append(data[j][len(data[j])-1])

title2 = title[0:14]

idx = []

for j in range(0, len(data2)):
    idx.append(data2[j].pop(len(data2[j])-1))

principalDf = pd.DataFrame(data = data2, columns = title2)
idxDf = pd.DataFrame(idx, columns = ['targets'])
finalDf = pd.concat([principalDf, idxDf], axis = 1)

print(finalDf)

seaborn.pairplot(finalDf, hue = 'targets')

print("Ready to plot")

plt.show()

