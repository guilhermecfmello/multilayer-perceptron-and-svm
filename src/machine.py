import numpy as np
import os.path
import csv

class machine:

    def getDatasetTraining(self, dataName):
        trainingName = dataName + '_training.csv'
        trainingNumpy = dataName + '_training'
        xTrain = []
        yTrain = []
        if not os.path.isfile(trainingNumpy+'_X.npy') and not os.path.isfile(trainingNumpy+'_Y.npy'):
            print('Geting X and Y from csv file and generating numpy optmizer...')
            digits=["zero","um","dois","tres","quatro","cinco","seis","sete","oito","nove"]
            with open(trainingName) as dataset:
                reader = csv.reader(dataset, delimiter=',')
                for row in reader:
                    xTrain.append(row[:-1])
                    y = np.zeros(10,dtype=float)
                    y[digits.index(row[-1:][0])] = 1
                    yTrain.append(y)
            np.save(trainingNumpy+'_X', xTrain)
            np.save(trainingNumpy+'_Y', yTrain)
        else:
            print('Enjoying X and Y numpy format file pre-saved...')
            xTrain = np.load(trainingNumpy+'_X.npy')
            yTrain = np.load(trainingNumpy+'_Y.npy')
        print('getDatasetTraining finished\n')
        return xTrain, yTrain
    
    def getDatasetValidation(self, dataName):
        trainingName = dataName + '_validation.csv'
        trainingNumpy = dataName + '_validation'
        x = []
        y = []
        if not os.path.isfile(trainingNumpy+'_X.npy') and not os.path.isfile(trainingNumpy+'_Y.npy'):
            print('Geting X and Y from csv file and generating numpy optmizer...')
            digits=["zero","um","dois","tres","quatro","cinco","seis","sete","oito","nove"]
            with open(trainingName) as dataset:
                reader = csv.reader(dataset, delimiter=',')
                for row in reader:
                    x.append(row[:-1])
                    yAux = np.zeros(10,dtype=float)
                    yAux[digits.index(row[-1:][0])] = 1
                    y.append(yAux)
            np.save(trainingNumpy+'_X', x)
            np.save(trainingNumpy+'_Y', y)
        else:
            print('Enjoying X and Y numpy format file pre-saved...')
            x = np.load(trainingNumpy+'_X.npy')
            y = np.load(trainingNumpy+'_Y.npy')
        print('getDatasetValidation finished\n')
        return x, y

    def getDatasetTest(self, dataName):
        trainingName = dataName + '_test.csv'
        trainingNumpy = dataName + '_test'
        x = []
        y = []
        if not os.path.isfile(trainingNumpy+'_X.npy') and not os.path.isfile(trainingNumpy+'_Y.npy'):
            print('Geting X and Y from csv file and generating numpy optmizer...')
            digits=["zero","um","dois","tres","quatro","cinco","seis","sete","oito","nove"]
            with open(trainingName) as dataset:
                reader = csv.reader(dataset, delimiter=',')
                for row in reader:
                    x.append(row[:-1])
                    yAux = np.zeros(10,dtype=float)
                    yAux[digits.index(row[-1:][0])] = 1
                    y.append(yAux)
            np.save(trainingNumpy+'_X', x)
            np.save(trainingNumpy+'_Y', y)
        else:
            print('Enjoying X and Y numpy format file pre-saved...')
            x = np.load(trainingNumpy+'_X.npy')
            y = np.load(trainingNumpy+'_Y.npy')
        print('getDatasetTest finished\n')
        return x, y