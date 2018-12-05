import numpy as np
import numpy.matlib
from sklearn.preprocessing import normalize

class transMatrix:

    def __init__(self, sList, sprays, shape):
        if len(sList) != len(sprays):
            print ("data error 1")
            return
        self.tmJun2Jul = np.matlib.zeros((shape[0], shape[1]))
        self.tmJul2Aug = np.matlib.zeros((shape[0], shape[1]))
        self.stmJun2Jul = np.matlib.zeros((shape[0], shape[1]))
        self.stmJul2Aug = np.matlib.zeros((shape[0], shape[1]))
        self.generateMatrix(sList, sprays)

    def generateMatrix(self, sList, sprays):
        for i in range(len(sList)):
            sps = sprays[i]
            sts = sList[i]
            if sps[0] == 0:
                self.tmJun2Jul[sts[0], sts[1]] += 1
            else:
                self.stmJun2Jul[sts[0], sts[1]] += 1
            if sps[1] == 0:
                self.tmJul2Aug[sts[1], sts[2]] += 1
            else:
                self.stmJul2Aug[sts[1], sts[2]] += 1
        #for states in sList:
        #    self.tmJun2Jul[states[0], states[1]] += 1
        #    self.tmJul2Aug[states[1], states[2]] += 1
        self.tmJun2Jul = normalize(self.tmJun2Jul, axis=1, norm='l1')
        self.tmJul2Aug = normalize(self.tmJul2Aug, axis=1, norm='l1')
        self.stmJun2Jul = normalize(self.stmJun2Jul, axis=1, norm='l1')
        self.stmJul2Aug = normalize(self.stmJul2Aug, axis=1, norm='l1')









