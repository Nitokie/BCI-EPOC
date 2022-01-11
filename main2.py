import csv
import torch
from torch import nn
from torch.utils.data import DataLoader
import random
from torch.utils.data.dataset import random_split
import time
import statistics
from numpy.fft import fft

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        #n_in, n_h, n_out = 30, 15, 7
        #n_in, n_h, n_h2, n_out = 44, 30, 20, 7
        n_in, n_h, n_h2, n_out = 14, 11, 9, 7
        #n_in, n_h, n_h2, n_out = 30, 20, 10, 7
        self.linear = nn.Sequential(
            nn.Linear(n_in, n_h),
            nn.ReLU(inplace=True),
            nn.Linear(n_h, n_h2),
            nn.ReLU(inplace=True),
            nn.Linear(n_h2, n_out),
            nn.Softmax(dim=0)
        )

    def forward(self, x):
        return self.linear(x)

#Carre rouge, Losange bleu, Rond vert, Etoile jaune, Triangle violet 

def readcsv():
    #fichiers = ["Carre rouge_EPOCPLUS_139054_2021.10.11T10.26.21+02.00.md.mc.pm.bp.csv", "Losange bleu_EPOCPLUS_118831_2021.11.08T11.22.57+01.00.md.mc.pm.bp.csv", "Rond vert_EPOCPLUS_118831_2021.11.08T11.25.46+01.00.md.mc.pm.bp.csv", "Etoile jaune_EPOCPLUS_139054_2021.12.14T13.53.37+01.00.md.mc.pm.bp.csv", "Triangle violet_EPOCPLUS_139054_2021.12.14T13.56.20+01.00.md.mc.pm.bp.csv"]
    #fichiers = ["JMCarreRouge_EPOCPLUS_146045_2022.01.05T20.41.19+01.00.md.mc.pm.bp.csv", "JMTriangleBleu_EPOCPLUS_146045_2022.01.05T20.45.00+01.00.md.mc.pm.bp.csv", "JMRondVert_EPOCPLUS_146045_2022.01.05T20.43.10+01.00.md.mc.pm.bp.csv"]
    fichiers = ["Avancer_EPOCPLUS_139054_2022.01.11T11.06.20+01.00.md.mc.pm.bp.csv", "Reculer_EPOCPLUS_139054_2022.01.11T11.08.08+01.00.md.mc.pm.bp.csv", "Gauche_EPOCPLUS_139054_2022.01.11T11.09.27+01.00.md.mc.pm.bp.csv", "Droite_EPOCPLUS_139054_2022.01.11T11.11.16+01.00.md.mc.pm.bp.csv"]
    
    data = []
    dataFFT = []
    title = []
    for f in range(0, len(fichiers)):
        eegTime = []
        with open(fichiers[f], "r") as file:
            text = csv.reader(file)
            i = 0
            for lines in text:
                if i == 1 and f == 0:
                    #print(lines[3:18], lines[24:39])
                    print(lines[4:19], lines[25:40])
                    columns = lines[4:19]
                    
                    #columns = lines[3:18] + lines[24:39]
                    for k in range(0, len(columns)):
                        title.append(columns[k])
                elif i > 2:

                    eegTimeStep = lines[4:18]
                    #eegTimeStep = lines[3:17]
                    step = lines[4:19]
                    #step = lines[3:18] + lines[24:39]
                    step.append(f+1)
                    try:
                        for j in range(0, len(step)):
                            step[j] = float(step[j])
                        data.append(step)
                    except ValueError:
                        print(step)
                    for j in range(0, len(eegTimeStep)):
                        eegTimeStep[j] = float(eegTimeStep[j])
                    eegTime.append(eegTimeStep)
                i += 1
        #fft
        dataFFT.append(fastFourierT(eegTime))
    print(len(data[0]))
    print(len(data))
    dataFFTttl = []
    for i in range(0, len(dataFFT)):
        for j in range(0, len(dataFFT[i])):
            dataFFTttl.append(dataFFT[i][j])

    for i in range(0, len(dataFFTttl)):
        verif = data[i].pop(len(data[i])-1)
        for j in range(0, len(dataFFTttl[i])):
            data[i].append(dataFFTttl[i][j])
        data[i].append(verif)

    random.shuffle(data)
#    print(len(data[0]))
#    print(data[0])
#    data2 = []
#    for i in range(0, len(data)):
#        data2.append(data[i][30:46])
    return data, title
#    return data2

def fastFourierT(dataTime):
    #from columns to rows
    dataTimeR = []
    for i in range(0, len(dataTime[0])):
        dataTimeR.append(colToRow(dataTime, i))
    #substracting mean
    dataTimeS = []
    for i in range(0, len(dataTimeR)):
        dataTimeS.append(subtractMean(dataTimeR[i]))
    #fft
    dataFFTR = []
    for i in range(0, len(dataTimeS)):
        dataFFTR.append(fft(dataTimeS[i]))
    #from rows to col
    dataFFT = []
    for i in range(0, len(dataFFTR[0])):
        dataFFT.append(colToRow(dataFFTR, i))
    #return fft
    return dataFFT

def colToRow(liste, idx):
    listeRow = []
    for i in range(0, len(liste)):
        listeRow.append(liste[i][idx])
    return listeRow

def subtractMean(liste):
    #print(liste)
    meanListe = statistics.mean(liste)
    for i in range(0, len(liste)):
        liste[i] = liste[i] - meanListe
    return liste

def convert(data, model, criterion, optimizer, device, listeY):
    valData = []
    listeValY = []
    accuTraining = []
    accuVal = []
    for j in range(0, len(data)):
        if j < 0.4*len(data) or j > 0.6*len(data):
            ptTensor = torch.FloatTensor(data[j]).to(device)
            ptY = torch.FloatTensor(listeY[j]).to(device)
            ptTensor.requires_grad_()
            ptY.requires_grad_()
            model, criterion, optimizer, accuracy = training(ptTensor, ptY, model, criterion, optimizer)
            accuTraining.append(accuracy)
        else:
            valData.append(data[j])
            listeValY.append(listeY[j])
    with open('log.txt', 'a') as log:
        log.write('\n\nVALIDATION STEP\n\n')
    for i in range(0, len(valData)):
        ptVal = torch.FloatTensor(valData[i]).to(device)
        ptValY = torch.FloatTensor(listeValY[i]).to(device)
        model, criterion, accuracy = eval(ptVal, ptValY, model, criterion)
        accuVal.append(accuracy)
    return model, criterion, optimizer, accuTraining, accuVal

def part(data):
    newData = []
    for i in range(0, len(data)):
        if i > 0.2*len(data) and i < 0.8*len(data):
            newData.append(data[i])
    return newData

def predi(data):
    predict = []
    for j in range(0, len(data)):
        result = data[j].pop(len(data[j])-1)
        if result == 1.0:
            predict.append([0, 1.0, 0, 0, 0, 0, 0])
        elif result == 2.0:
            predict.append([0, 0, 1.0, 0, 0, 0, 0])
        elif result == 3.0:
            predict.append([0, 0, 0, 1.0, 0, 0, 0])
    return predict

def maxListe(liste):
    maxi = 0
    for j in range(0, len(liste)):
        maxT = max(liste[j])
        if maxT > maxi:
            maxi = maxT
    return maxi

def normal(liste, max):
    for j in range(0, len(liste)):
        if len(liste[j]) > 1:
            for i in range(0, len(liste[j])):
                liste[j][i] = liste[j][i]/max
        else:
            liste[j] = liste[j]/max
    return liste

def training(ptTensor, ptY, model, criterion, optimizer):
    correct = 0
    for epoch in range(50):
        model.train()
        print(len(ptTensor))
        y_pred = model(ptTensor)
        y_pred_liste = reverseConvert(y_pred)
        y_pred = probaSoft(y_pred_liste)
        loss = criterion(y_pred, ptY)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        correct += (y_pred == ptY).float().sum()
        with open('log.txt', 'a') as log:
            log.write('Epoch: {}, Loss: {}\n'.format(epoch, loss.item()))
    accuracy = 100*correct/len(ptTensor)
    with open('log.txt', 'a') as log:
        log.write("Accuracy : {}\n\n".format(accuracy))
    return model, criterion, optimizer, accuracy

def eval(ptVal, ptValY, model, criterion):
    correct = 0
    for epoch in range(50):
        with torch.no_grad():
            model.eval()
            y_pred = model(ptVal)
            y_pred_liste = reverseConvert(y_pred)
            y_pred = probaSoft(y_pred_liste)
            val_loss = criterion(y_pred, ptValY)
            correct += (y_pred == ptValY).float().sum()
            with open('log.txt', 'a') as log:
                log.write('Epoch: {}, Loss: {}\n'.format(epoch, val_loss.item()))
    accuracy = 100*correct/len(ptVal)
    with open('log.txt', 'a') as log:
        log.write("Accuracy : {}\n\n".format(accuracy))
    return model, criterion, accuracy

def approxLinear(liste):
    print(liste)
    for i in range(0, len(liste)):
        if liste[i] < 0.6:
            liste[i] = 0
        elif liste[i] > 0.5 and liste[i] <= 1:
            liste[i] = 1
        else:
            print("Pas dans intervalle")
    liste = torch.FloatTensor(liste).to("cuda")
    return liste

def probaSoft(liste):
    print(liste)
    maxLocal = 0
    idx = 0
    for i in range(0, len(liste)):
        if liste[i] > maxLocal:
            maxLocal = liste[i]
            idx = i
    for i in range(0, len(liste)):
        if i != idx:
            liste[i] = 0
        else:
            liste[i] = 1
    liste = torch.FloatTensor(liste).to("cuda")
    return liste

def reverseConvert(ptListe):
    liste = []
    for i in range(0, len(ptListe)):
        liste.append(ptListe[i].tolist())
    return liste

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    with open('log.txt', 'w') as log:
        log.write('Timestamp : {}\n'.format(time.time()))
        log.write("Using {} device".format(device))
    #n_out : 0, 1, 2, 3, 4, 5, 6 -> 6 dir + 1 unknonws case
    #n_out : 1 -> square; 2 -> losange; 3 -> circle
    model = NeuralNetwork().to(device)
    criterion = torch.nn.MSELoss()
    #Lower learning rate
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-100)
    trueData, title = readcsv()
    data = part(trueData)
    listeY = predi(data)
    maxi = maxListe(data)
    data = normal(data, maxi)
    model, criterion, optimizer, accuTraining, accuVal = convert(data, model, criterion, optimizer, device, listeY)
    accuTraining = reverseConvert(accuTraining)
    accuVal = reverseConvert(accuVal)
    print("Accuracy:\nTraining: min: {}, max: {}, mean: {}\nValidation: min: {}, max: {}, mean: {}".format(min(accuTraining), max(accuTraining), statistics.mean(accuTraining), min(accuVal), max(accuVal), statistics.mean(accuVal)))
    with open('log.txt', 'a') as log:
        log.write("\nAccuracy:\nTraining: min: {}, max: {}, mean: {}\nValidation: min: {}, max: {}, mean: {}".format(min(accuTraining), max(accuTraining), statistics.mean(accuTraining), min(accuVal), max(accuVal), statistics.mean(accuVal)))


if __name__=='__main__':
    main()
