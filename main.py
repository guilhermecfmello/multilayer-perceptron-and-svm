from svm import svm
import sys
import utils


if __name__ == "__main__":
    dataName = utils.getArgs(sys.argv, '-i')
    if dataName: print('opening dataset: data/' + dataName)
