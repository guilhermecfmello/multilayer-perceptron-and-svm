import machine
import numpy as np
# from joblib import dump, load
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
        x_test, y_test = super().getDatasetTest(self.dataPath)
        c = 0.1
        gamma = 0.1
        s = svm.SVC(C=c, kernel='linear', gamma=gamma)
        y_train2 = self.__vectorToScalar(y_train)
        s.fit(x_train, y_train2)
        y_predict = s.predict(x_test)
        y_predict = self.__scalarToVector(y_predict)
        self.y_predict = y_predict
        self.y_expect = y_test
        return True
        # print(y_predict)
        # exit()

    def evalMetrics(self):
        print("Evaluating SVM metrics...")

        expect = self.__vectorToScalar(self.y_expect)
        predict = self.__vectorToScalar(self.y_predict)
        print(expect[:30])
        print(predict[:30])

        # vp, vn, fp, fn
        VP, VN, FP, FN = 0, 1, 2, 3
        metrics = np.zeros((10,4))
        for p, e in zip(predict, expect):
            # Right prediction, true (verdadeiro alguma coisa)
            if p==e:
                metrics[e][VP] = metrics[e][VP] + 1
                metrics[:e][VN] = metrics[:e][VN] + 1
                metrics[e+1:][VN] = metrics[e+1:][VN] + 1
            # Wrong prediction, false (falso alguma coisa)
            else:
                metrics[e][FN] = metrics[e][FN] + 1
                metrics[p][FP] = metrics[p][FP] + 1
                for i in range(len(metrics)):
                    if i != e and i != p:
                        metrics[i][VN] = metrics[i][VN] + 1
                
                       