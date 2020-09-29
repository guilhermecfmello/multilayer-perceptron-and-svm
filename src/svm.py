import machine
import pickle
import os.path
import numpy as np
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

    def rbf_training(self, c=0.1, gamma=0.1, k='rbf'):
        self.c = c
        self.gamma = gamma
        self.kernel = k
        fileName=DATA_FOLDER+'svm_c'+str(c)+'_gamma'+str(gamma)+'_kernel'+str(k)+'.sav'
        print("Runing SVM RBF training...")
        x_train, y_train = super().getDatasetTraining(self.dataPath)
        x_test, y_test = super().getDatasetTest(self.dataPath)

        # if exists svm saved
        if not os.path.isfile(fileName):
            print('There is not an svm instance saved with this parameters, generating from scretch...')
            # decision_function_shape: [ovo=n*(n-1)/2,ovr=n]
            s = svm.SVC(C=c, kernel=k, gamma=gamma, decision_function_shape='ovo')
            y_train2 = self.__vectorToScalar(y_train)
            s.fit(x_train, y_train2)
            print('dumping svm on: ' + fileName)
            pickle.dump(s, open(fileName, 'wb'))
        else:
            print('loading svm from disk on: '+fileName)
            s = pickle.load(open(fileName, 'rb'))
        
        y_predict = s.predict(x_test)
        y_predict = self.__scalarToVector(y_predict)
        self.y_predict = y_predict
        self.y_expect = y_test
        return True

    def evalMetrics(self):
        print("Evaluating SVM metrics...")
        
        expect = self.__vectorToScalar(self.y_expect)
        predict = self.__vectorToScalar(self.y_predict)

        # vp, vn, fp, fn
        VP, VN, FP, FN = 0, 1, 2, 3
        metrics = np.zeros((10,4))
        k = 0
        for p, e in zip(predict, expect):
            e_int = int(e)
            p_int = int(p)
            # Right prediction, true (verdadeiro alguma coisa)
            if p==e:
                metrics[e_int][VP] = metrics[e_int][VP] + 1
                metrics.T[VN][:e_int] = metrics.T[VN][:e_int] + 1
                metrics.T[VN][e_int+1:] = metrics.T[VN][e_int+1:] + 1
            # Wrong prediction, false (falso alguma coisa)
            else:
                k = k + 1
                metrics[e_int][FN] = metrics[e_int][FN] + 1
                metrics[p_int][FP] = metrics[p_int][FP] + 1
                mi = min([e_int, p_int])
                ma = max([e_int, p_int])
                metrics[:mi].T[VN:VN+1] = metrics[:mi].T[VN:VN+1] + 1
                metrics[mi+1:ma].T[VN:VN+1] = metrics[mi+1:ma].T[VN:VN+1] + 1
                metrics[ma+1:].T[VN:VN+1] = metrics[ma+1:].T[VN:VN+1] + 1
        for i in range(10):
            if metrics[i].sum() != len(self.y_expect):
                print("Metrics inconsistent")
        
        print('========== METRICS ==========')
        print('SVM Configs: [c='+str(self.c)+', gamma='+str(self.gamma)+', kernel='+str(self.kernel)+']')
        print(metrics)
        for i in range(len(metrics)):
            vp = metrics[i][VP]
            vn = metrics[i][VN]
            fp = metrics[i][FP]
            fn = metrics[i][FN]
            precision = vp/(vp+fp) if (vp+fp) > 0 else 0
            recall = vp/(vp+fn) if (vp+fn) > 0 else 0
            accuracy = (vp+vn)/(vp+fp+fn+vn)
            f1 = 2*(precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
            print('For class='+str(i))
            print('\tPrecision: '+str(precision))
            print('\tRecall: '+str(recall))
            print('\tAccuracy: '+str(accuracy))
            print('\tF-1: '+str(f1))
        return True