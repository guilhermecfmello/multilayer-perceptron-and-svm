import machine
import numpy as np
from joblib import dump, load
from sklearn import svm
from sklearn.model_selection import train_test_split

DATA_FOLDER = '../data/'

class SVM(machine.machine):
    
    def __init__(self, dataName):
        self.dataPath = DATA_FOLDER + dataName
    
    def __vectorToScalar(self, y_some):
        newY = np.array([])
        for y in y_some:
            newY = np.append(newY, np.where(y == 1))
        return newY
            
    def __scalarToVector(self, y_some):
        newY = []
        for y in y_some:
            yAux = np.zeros(10,dtype=float)
            yAux[int(y)] = 1
            newY.append(yAux)
        return newY

    def rbf_training(self):
        print("Runing SVM RBF training...")
        x_train, y_train = super().getDatasetTraining(self.dataPath)
        c = 0.1
        gamma = 0.1
        s = svm.SVC(C=c, kernel='linear', gamma=gamma)
        y_train2 = self.__vectorToScalar(y_train)
        s.fit(x_train, y_train2)
        x_test, y_test = super().getDatasetTest(self.dataPath)
        y_predict = s.predict(x_test)
        y_predict = self.__scalarToVector(y_predict)
        
        