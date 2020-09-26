from svm import SVM
import sys
import utils


if __name__ == "__main__":
    dataName = utils.getArgs(sys.argv, '-i')
    # if dataName: print('opening dataset: data/' + dataName)
    svm = SVM(dataName)