from sklearn import svm
import machine
import numpy as np
from sklearn.model_selection import train_test_split
import csv

DATA_FOLDER = '../data/'

class SVM(machine.machine):
    
    def __init__(self, dataName):
        self.dataPath = DATA_FOLDER + dataName
        x_train, y_train = super().getDatasetTraining(self.dataPath)
        # x_valid, y_valid = super().getDatasetValidation(self.dataPath)
        # x_test, y_test = super().getDatasetTest(self.dataPath)
        