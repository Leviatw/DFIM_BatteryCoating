import numpy as np
from scipy.stats import matrix_normal, norm
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

def ewmaonlineType2(L,mu0,Phi,rho,w,p,DistType,CovType,B,Bs,lambda1,allgamma, isewma=False, maxIter = 2, mT2 = 0, sd = 1, initial = 0):
    Covw = getCov(rho,w,CovType)
    Covp = getCov(rho,p,CovType)

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

    for t in range(nT):
        dall = [[] for i in range(gammalength)]
        thetai = [[] for i in range(gammalength)]
        if DistType == "Normal":
            eps = matrix_normal.rvs(mean=np.zeros([w,p]), rowcov=Covw, colcov=Covp)
            y = np.array(mu0+eps) 
        elif DistType == "Exponential":
            eps = matrix_normal.rvs(mean=np.zeros([w,p]), rowcov=Covw, colcov=Covp)
            y = np.array(mu0+(-1 * np.log(1 - norm.cdf(eps))))  
            
        for i in range(gammalength):
            if t==0:
                Yhat = H[1]@y@H[2]
            else:
                Snow = 0
                iiter = 0
                thetaold = Yhat
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

    return t

# total 4 files
# [T2, CovType, phi]
parser = argparse.ArgumentParser()
parser.add_argument('inputfile', help='Specify an input file', type=int)
args = parser.parse_args()
index = args.inputfile

filename = '/home/tgong33/ImageMonitoring/InCtrl/Type2/in/input%d.dat' %index
my_shelf = shelve.open(filename)
globals()['T2']=my_shelf['T2']
globals()['CovType']=my_shelf['CovType']
globals()['DistType']=my_shelf['DistType']
my_shelf.close()
del my_shelf
mT2,sd,Ttr,Itr = chartIC(T2)
mT2 = mT2[:,0]
sd = sd[:,0]
L = np.quantile(Ttr, 0.999)

ControlType = 'InControl'
DataType = 'Image'
phi = 0.3
rho = 0.3
w = 100
p = 200

Tag = (ControlType,DataType,CovType,DistType,'phi',str(phi),'rho',str(rho))
TagJoin = ''.join(Tag)

allgamma = [1e-2,1,2,3,5,9,11]
lambda1 = [0.9,0.1,0.1]
nx = w
ny = p
kx = 10; ky = 30; kt = 3
sdx = 3; sdy = 3; sdt = 3
skx = round(nx/2)
sky = round(nx/2)
Phi = phi * np.eye(p)
mu0 = 5*np.ones([w,p])
seedid = 19980422

InControlList = []
for n in tqdm(range(TstSize)):
    np.random.seed(seedid + n)
    B = [bsplineBasis(nx,kx,sdx),  bsplineBasis(ny,ky,sdy)]
    Bs = [bsplineBasis(nx,skx,1), bsplineBasis(ny,sky,1)]           
    t = ewmaonlineType2(L,mu0,Phi,rho,w,p,DistType,CovType,B,Bs,lambda1,allgamma,maxIter=3,mT2=mT2,sd=sd)
    InControlList.append(t)
    
filename = "/home/tgong33/ImageMonitoring/InCtrl/Type2/out/output%d.npy" %index
np.save(filename,InControlList)

print(InControlList)
print('Setting %d' %index)
print('Exit code 0')