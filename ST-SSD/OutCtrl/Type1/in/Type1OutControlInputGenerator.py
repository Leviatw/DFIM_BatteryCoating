# Type 1
import shelve
import numpy as np
CovList = ['TriDiagonal','Exponential']
phiList = [0.3, 0.7]
ShiftList = ['Sparse','Stepwise','Zigzag']
normList = [[2,5],[1,2],[2,5]]
i = 0
for CovType in CovList:
    for phi in phiList:
        for sft in range(3):
            ShiftType = ShiftList[sft]
            subnormList = normList[sft]
            for deltanorm in subnormList:
                        ControlType = 'InControl'
                        DataType = 'Profile'
                        DistType = 'Normal'
                        rho = 0.3

                        Tag = ('InControl',DataType,CovType,DistType,'phi',str(phi),'rho',str(rho))
                        TagJoin = ''.join(Tag)
                        LTag = ('OutControlProfile',CovType,'phi',str(phi),ShiftType,'norm',str(deltanorm))
                        LTagJoin = ''.join(LTag)
                        filename = "/home/tgong33/ImageMonitoring/OutCtrl/Type1/in/input%s.dat" % i


                        T2 = np.load("/home/tgong33/ImageMonitoring/InCtrl/Type1/in/TestStat"+TagJoin+".npy")

                        my_shelf = shelve.open(filename,'n')
                        my_shelf['T2'] = globals()['T2']
                        my_shelf['CovType'] = globals()['CovType']
                        my_shelf['phi'] = globals()['phi']
                        my_shelf['ShiftType'] = globals()['ShiftType']
                        my_shelf['deltanorm'] = globals()['deltanorm']
                        my_shelf['Setting'] = globals()['LTagJoin']
                        my_shelf.close()
                        i = i+1