import csv
import statistics
from numpy.fft import fft
import numpy as np
from main2 import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd

dataTrue, title = readcsv()
data = part(dataTrue)
data2 = []
datafft = []

##for i in range(0, len(data)):
##    data2.append(data[i][0:30])
##    data2[i].append(data[i][44])
##    datafft.append(data[i][30:44])

for i in range(0, len(data)):
    data2.append(data[i][0:15])
    data2[i].append(data[i][len(data[i])-1])

##for i in range(0, len(data)):
##    data2.append(data[i][0:1])
##    data2[i].append(data[i][44])
##    title = title[0:1]


##datafftpart = []
##
##for i in range(0, len(datafft)):
##    dstep = []
##    for j in range(0, len(datafft[i])):
##        dstr = datafft[i][j]
##        dstep.append(dstr.real)
##        dstep.append(dstr.imag)
##    dstep.append(data[i][44])
##    datafftpart.append(dstep)
    
##data2 = datafftpart

##Title only eeg
##title = title[0:15]

##Title FFT
#title = ["AF3.real", "AF3.im", "F7.real", "F7.im", "F3.real", "F3.im", "FC5.real", "FC5.im", "T7.real", "T7.im", "P7.real", "P7.im", "O1.real", "O1.im", "O2.real", "O2.im", "P8.real", "P8.im", "T8.real", "T8.im", "FC6.real", "FC6.im", "F4.real", "F4.im", "F8.real", "F8.im", "AF4.real", "AF4.im"]


idx = []

for i in range(0, len(data2)):
    idx.append(data2[i].pop(len(data2[i])-1))

data2 = StandardScaler().fit_transform(data2)

pca = PCA(n_components = 2)

principalComponents = pca.fit_transform(data2)

principalDf = pd.DataFrame(data = principalComponents, columns = ["principal component 1", "principal component 2"])

idxDf = pd.DataFrame(idx, columns = ['targets'])

finalDf = pd.concat([principalDf, idxDf], axis = 1)

print(pca.explained_variance_ratio_)

fig = plt.figure(figsize = (8, 8))
ax = fig.add_subplot(2, 1, 1)
ax.set_xlabel("Principal Component 1", fontsize = 15)
ax.set_ylabel("Principal Component 2", fontsize = 15)
ax.set_title("2 component PCA" + "\n" + str(pca.explained_variance_ratio_), fontsize = 20)

targets = [1.0, 2.0, 3.0, 4.0, 5.0]
colors = ["r", "b", "g", "yellow", "purple"]
for target, color in zip(targets, colors):
    indicesToKeep = finalDf["targets"] == target
    ax.scatter(finalDf.loc[indicesToKeep, "principal component 1"], finalDf.loc[indicesToKeep, "principal component 2"], c = color, s = 50)

ax.legend(targets)
ax.grid()


#Correlation circle
eigval = pca.explained_variance_
eigval = (len(data2)-2)/(len(data2)-1)*eigval

sqrt_eigval = np.sqrt(eigval)
corvar = np.zeros((len(title), len(title)))
##print(corvar)
##print(sqrt_eigval[0])
##print(pca.components_)
##print(corvar[:, 2])
##print(pca.components_[2, :])
##print(sqrt_eigval[2])
print(len(pca.components_))
print(len(sqrt_eigval))
for k in range(2):
    corvar[:, k] = pca.components_[k, :] * sqrt_eigval[k]

##fig, ax = plt.subplots()
ay = fig.add_subplot(2, 1, 2)
an = np.linspace(0, 2 * np.pi, 100)
ay.plot(np.cos(an), np.sin(an), 'b', linewidth=0.5)

print(len(title))

for i in range(0, corvar.shape[0]):
    ay.arrow(0,
             0,
             corvar[i, 0],
             corvar[i, 1],
             head_width = 0.1,
             head_length = 0,
             color = 'r')
    ay.text(corvar[i, 0]*1.15, corvar[i, 1]*1.15, title[i],
            color = 'k', ha = 'center', va = 'center')

ay.axis('equal')
ay.set_xlabel('DIM1', fontsize = 11)
ay.set_ylabel('DIM2', fontsize = 11)
ay.set_title('Correlation circle')

fig.show()
