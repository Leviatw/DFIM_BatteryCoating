# Type 1
import shelve
import numpy as np
DistList = ['Normal','Exponential']
CovList = ['TriDiagonal','Exponential']
ShiftListImage = ['Sparse','Chessboard','Ring', 'Sine(row)', 'Sine(col)']
normList = [[10,20],[10,20],[20,30],[20,30],[20,30]]
i = 0
for DistType in DistList:
    for CovType in CovList:
        for sft in range(5):
            ShiftType = ShiftListImage[sft]
            subnormList = normList[sft]
            for DeltaNorm in subnormList:
                        DataType = 'Image'
                        phi = 0.3
                        rho = 0.3

                        Tag = ('InControl',DataType,CovType,DistType,'phi',str(phi),'rho',str(rho))
                        TagJoin = ''.join(Tag)
                        LTag = ('OutControlProfile',DistType, CovType, ShiftType,'norm',str(DeltaNorm))
                        LTagJoin = ''.join(LTag)
                        filename = "/home/tgong33/ImageMonitoring/OutCtrl/Type2/in/input%s.dat" % i


                        T2 = np.load("/home/tgong33/ImageMonitoring/InCtrl/Type2/in/TestStat"+TagJoin+".npy")

                        my_shelf = shelve.open(filename,'n')
                        my_shelf['T2'] = globals()['T2']
                        my_shelf['DistType'] = globals()['DistType']
                        my_shelf['CovType'] = globals()['CovType']
                        my_shelf['ShiftType'] = globals()['ShiftType']
                        my_shelf['DeltaNorm'] = globals()['DeltaNorm']
                        my_shelf['Setting'] = globals()['LTagJoin']
                        my_shelf.close()
                        i = i+1