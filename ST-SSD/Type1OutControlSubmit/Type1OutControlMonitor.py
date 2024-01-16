import numpy as np
from scipy.stats import multivariate_normal
import scipy.interpolate as spl
from tqdm import tqdm
import shelve
import argparse
TstSize = 1000
def getCov(rho, p, type):
    CovList = ['TriDiagonal','Exponential']
    Sigma = np.eye(p)
    if type == CovList[0]:
        tmp = np.ones(p-1)*rho
        Sigma = Sigma+np.diag(tmp, k = 1)+np.diag(tmp, k = -1)
    elif type == CovList[1]:
        for d in range(1, p-1):
            tmp = np.ones(p-d)*(rho**d)
            Sigma = Sigma+np.diag(tmp, k = d)+np.diag(tmp, k = -d)
    return Sigma

def bsplineBasis(n,k,deg):
    knots = np.r_[np.zeros(deg),np.linspace(0,n-1,k),(n-1) * np.ones(deg)]
    x = np.arange(n)
    m = len(knots) - deg - 1
    v = np.zeros((m, len(x)))
    d = np.eye(m, len(knots))
    for i in range(m):
        v[i] = spl.splev(x, (knots, d[i], deg))
    return v.T

def chartOC( T2tr,mT2,sd):
    T2c = T2tr - mT2
    T2tn = T2c/sd
    Itr = np.argmax(T2tn,axis=0);
    Ttr = np.max(T2tn,0)
    return Ttr,Itr

def chartIC( T2tr,ifIncludeNAN = True, ndel = 0):
    T2trd= T2tr[:,ndel:];
    if ifIncludeNAN:
        mT2 = np.nanmean(T2trd,axis=1);
        sd = np.nanstd(T2trd,axis=1);
    else:
        mT2 = np.mean(T2trd,axis=1);
        sd = np.std(T2trd,axis=1);
    mT2 = mT2.reshape(-1,1)
    sd = sd.reshape(-1,1)
    
    T2c = T2trd - mT2
    T2tn = T2c/sd
    Itr = np.argmax(T2tn,axis=0);
    Ttr = np.max(T2tn,0)    
    return mT2,sd,Ttr,Itr

def ewmaonlineType1outofcontrol(deltanorm,ShiftType,L,mu0,Phi,rho,w,p,CovType,B,Bs,lambda1,allgamma, isewma=False, maxIter = 2, mT2 = 0, sd = 1, initial = 0):
    Cov = getCov(rho,p,CovType)

    lambdat = lambda1[0]
    lambdaxy = lambda1[1:]
    lambdat1 = lambdat
    ndim = 3
    issave = 1

    gammalength = len(allgamma)
    softthreshold = lambda residual,gamma : np.sign(residual)*np.maximum(np.abs(residual) - gamma, 0)
    B.insert(0,[])
    constructD =lambda n: np.diff(np.eye(n),1,axis=0);
    nT = 5000

    LL = 2*np.linalg.norm(Bs[0],ord=2)**2*np.linalg.norm(Bs[1],ord=2)**2
    X = np.zeros((Bs[0].shape[1],Bs[1].shape[1]))
    BetaS = np.zeros((Bs[0].shape[1],Bs[1].shape[1]))

    D = [[] for i in range(ndim)];
    H = [[] for i in range(ndim)];

    for idim in range(1,ndim):
        D[idim] = constructD(B[idim].shape[1]);
        H[idim] = B[idim]@ np.linalg.solve(B[idim].T@B[idim] + lambdaxy[idim-1] * (D[idim].T@D[idim]),B[idim].T);
    
    T2 = np.zeros((gammalength,nT))
    Tte = np.zeros(nT)
    Itr = np.zeros(nT,dtype=int)
    tnew = 1

    if ShiftType == 'Sparse':
        shift_basis = np.zeros(p) # sparse
        shift_basis[18:23] = np.ones(5)
        shift_basis = deltanorm/np.sqrt(np.sum(shift_basis**2))*shift_basis

    elif ShiftType == 'Stepwise':
        shift_basis = np.zeros(p) # piece_1
        shift_basis[50:100] = np.ones(50)
        shift_basis[100:150] = 2 * np.ones(50)
        shift_basis[150:200] = 3 * np.ones(50)
        shift_basis = deltanorm/np.sqrt(np.sum(shift_basis**2))*shift_basis

    elif ShiftType == 'Zigzag':
        shift_basis = np.zeros(p) # zigzag
        left = 1
        right = -1
        for j in range(10):
            shift_basis[j*20:(j+1)*20] = np.linspace(left, right, 20)
            left, right = right, left
        shift_basis = deltanorm/np.sqrt(np.sum(shift_basis**2))*shift_basis        

    mu = mu0+shift_basis
    eps0 = multivariate_normal.rvs(np.zeros(p), Cov)
    data = []
    for i in range(w):
        eps = np.matmul(Phi, eps0) + multivariate_normal.rvs(np.zeros(p), Cov)
        data.append(mu + eps)
        eps0 = eps

    for t in range(nT):
        dall = [[] for i in range(gammalength)]
        thetai = [[] for i in range(gammalength)]
        y = np.array(data)
        for i in range(gammalength):
            if t==0:
                Yhat = H[1]@y@H[2]
            else:
                Snow = 0;
                iiter = 0;
                thetaold = Yhat;
                while iiter < maxIter:
                    iiter = iiter +1
                    BetaSold = BetaS
                    told = tnew

                    yhat = H[1]@(y-Snow)@H[2]
                    Yhat = lambdat1*yhat+(1-lambdat1)*thetaold

                    BetaSe = X + 2/LL* Bs[0].T@(y -Bs[0]@X@Bs[1].T - Yhat)@Bs[1]
                    BetaS = softthreshold(BetaSe,allgamma[i]/LL); 
                    Snow = Bs[0] @BetaS@ Bs[1].T
                    tnew = (1+np.sqrt(1+4*told**2))/2
                    if iiter==1:
                        X = BetaS
                    else:
                        X = BetaS+(told-1)/tnew*(BetaS-BetaSold)

                            

            BetaSe = BetaS + 2/LL*Bs[0].T@(y -Bs[0]@BetaS@Bs[1].T - Yhat)@Bs[1]
            BetaS = softthreshold(BetaSe,allgamma[i]/LL)
            Snow = Bs[0] @BetaS @ Bs[1].T
            d = Snow
            

            T2[i,t] = (np.sum(d*(y - Yhat)))**2/np.sum(d**2)
            dall[i] = d
            thetai[i] = Yhat
        

        if L:
            L1,Itr[t] = chartOC( T2[:,t],mT2,sd)
            Tte[t] = L1
            if L1 > L and t>initial:
                break

        eps = np.matmul(Phi, eps0) + multivariate_normal.rvs(np.zeros(p), Cov)
        data.append(mu + eps)
        eps0 = eps
        data = data[1:]

    return t

# total 24 (4*3*2) files
# [T2, CovType, phi, ShiftType, deltanorm]
parser = argparse.ArgumentParser()
parser.add_argument('inputfile', help='Specify an input file', type=int)
args = parser.parse_args()
index = args.inputfile

filename = '/home/tgong33/ImageMonitoring/OutCtrl/Type1/in/input%d.dat' %index
my_shelf = shelve.open(filename)
globals()['T2']=my_shelf['T2']
globals()['CovType']=my_shelf['CovType']
globals()['phi']=my_shelf['phi']
globals()['ShiftType'] = my_shelf['ShiftType']
globals()['deltanorm'] =  my_shelf['deltanorm'] 
globals()['Setting'] =  my_shelf['Setting'] 
my_shelf.close()
del my_shelf


mT2,sd,Ttr,Itr = chartIC(T2)
mT2 = mT2[:,0]
sd = sd[:,0]
L = np.quantile(Ttr, 0.999)

ControlType = 'OutControl'
DataType = 'Profile'
DistType = 'Normal'
rho = 0.3
w = 5
p = 200

allgamma = np.linspace(0.001,0.01,10)
lambda1 = [0.9,0.1,0.1]
nx = w
ny = p
kx = 3; ky = 40; kt = 3
sdx = 1; sdy = 3; sdt = 3
skx = round(nx/2)
sky = round(nx/2)

Phi = phi * np.eye(p)
mu0 = 5*np.ones(p)
seedid = 19990422

InControlList = []
for n in tqdm(range(TstSize)):
    np.random.seed(seedid + index*TstSize + n)
    B = [bsplineBasis(nx,kx,sdx),  bsplineBasis(ny,ky,sdy)]
    Bs = [bsplineBasis(nx,skx,1), bsplineBasis(ny,sky,1)]           
    t =ewmaonlineType1outofcontrol(deltanorm, ShiftType,L,mu0,Phi,rho,w,p,CovType,B,Bs,lambda1,allgamma,maxIter=3,mT2=mT2,sd=sd)
    InControlList.append(t)
    
filename = "/home/tgong33/ImageMonitoring/OutCtrl/Type1/out/output%d.npy" %index
np.save(filename,InControlList)

print(InControlList)
print('Setting %d' %index)
print(Setting)
print('Exit code 0')