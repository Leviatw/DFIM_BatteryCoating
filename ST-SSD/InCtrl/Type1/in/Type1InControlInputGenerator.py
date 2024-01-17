# Type 1
import shelve
import numpy as np
CovList = ['TriDiagonal','Exponential']
phiList = [0.3, 0.7]
i = 0
for CovType in CovList:
    for phi in phiList:
        ControlType = 'InControl'
        DataType = 'Profile'
        DistType = 'Normal'
        rho = 0.3

        Tag = (ControlType,DataType,CovType,DistType,'phi',str(phi),'rho',str(rho))
        TagJoin = ''.join(Tag)
        filename = "/home/tgong33/ImageMonitoring/InCtrl/Type1/in/input%s.dat" % i


        T2 = np.load("/home/tgong33/ImageMonitoring/InCtrl/Type1/in/TestStat"+TagJoin+".npy")

        my_shelf = shelve.open(filename,'n')
        my_shelf['T2'] = globals()['T2']
        my_shelf['CovType'] = globals()['CovType']
        my_shelf['phi'] = globals()['phi']
        my_shelf.close()
        i = i+1