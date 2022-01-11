##https://towardsdatascience.com/an-introduction-to-t-sne-with-python-example-5a3a293108d1
##https://towardsdatascience.com/t-sne-python-example-1ded9953f26

from main2 import *
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set(rc={'figure.figsize':(11.7, 8.27)})
#palette = sns.color_palette("bright", 5)
palette = sns.color_palette("bright", 4)

dataTrue, title = readcsv()
data = part(dataTrue)
data2 = []
datafft = []

##for i in range(0, len(data)):
##    data2.append(data[i][0:30])
##    data2[i].append(data[i][44])
##    datafft.append(data[i][30:44])

print(data[1])
for i in range(0, len(data)):
    data2.append(data[i][1:14])
    data2[i].append(data[i][len(data[i])-1])
print(data2[1])

title = title[1:14]

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
##    
##data2 = datafftpart

##title = ["AF3.real", "AF3.im", "F7.real", "F7.im", "F3.real", "F3.im", "FC5.real", "FC5.im", "T7.real", "T7.im", "P7.real", "P7.im", "O1.real", "O1.im", "O2.real", "O2.im", "P8.real", "P8.im", "T8.real", "T8.im", "FC6.real", "FC6.im", "F4.real", "F4.im", "F8.real", "F8.im", "AF4.real", "AF4.im"]

idx = []

for i in range(0, len(data2)):
    idx.append(data2[i].pop(len(data2[i])-1))

tsne = TSNE()
data_embedded = tsne.fit_transform(data2)
##print(tsne.explained_variance_ratio_)
sns.scatterplot(data_embedded[:, 0], data_embedded[:, 1], hue = idx, legend = 'full', palette = palette)
plt.title("T-SNE EEG")
plt.show()
