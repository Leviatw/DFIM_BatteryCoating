# Type 2
import shelve
import numpy as np
CovList = ['TriDiagonal','Exponential']
DistList = ['Normal','Exponential']
i = 0
for DistType in DistList:
    for CovType in CovList:
        ControlType = 'InControl'
        DataType = 'Image'
        phi = 0.3
        rho = 0.3

        Tag = (ControlType,DataType,CovType,DistType,'phi',str(phi),'rho',str(rho))
        TagJoin = ''.join(Tag)
        filename = "/home/tgong33/ImageMonitoring/InCtrl/Type2/in/input%s.dat" % i


        T2 = np.load("/home/tgong33/ImageMonitoring/InCtrl/Type2/in/TestStat"+TagJoin+".npy")

        my_shelf = shelve.open(filename,'n')
        my_shelf['T2'] = globals()['T2']
        my_shelf['CovType'] = globals()['CovType']
        my_shelf['DistType'] = globals()['DistType']
        my_shelf.close()
        i = i+1