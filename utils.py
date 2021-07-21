import numpy as np
from scipy.stats import ncx2,chi2,f
from scipy.special import logsumexp
def solve_pi(sig,delta,alpha,beta,N,T,Z,df = 1):
    # grid search for non-central parameter
    c_N = np.linspace(5,10,501)
    pdfs = ncx2.cdf(chi2.ppf(1-alpha,df),df,c_N)
    c_N = c_N[np.argmin(np.abs(pdfs-beta))]
    if df==1:
        s = T*np.square(delta)
    else:
        ZZ = np.array([np.matmul(z.reshape(-1,1),z[np.newaxis,:]) for z in Z])
        s = np.matmul(delta,np.matmul(np.sum(ZZ,axis=0),delta))
    d = np.square(sig)*c_N/N/s
    return 1/2-np.sqrt(1-4*d)/2,1/2+np.sqrt(1-4*d)/2


# calculate type I error p(reject H0 | H0)
def type_i_error(stats,alpha = 0.05, df = 1):
    #print (df)
    err = 1.*len(np.where(stats>=chi2.ppf(1-alpha,df))[0])/len(stats)
    #err = 1.*len(np.where(stats>=f.ppf(1-alpha,df,len(stats)-4))[0])/len(stats)
    return err, np.sqrt(err*(1-err)/len(stats))*2

# calculate power p(correctly reject H0 | H1)
def power(stats, alpha = 0.05, df = 1):
    power = 1.*len(np.where(stats>=chi2.ppf(1-alpha,df))[0])/len(stats)
    return power, np.sqrt(power*(1-power)/len(stats))*2

def regret_cal(opt,R):
    N = len(R)
    R = np.sum(np.vstack(R),axis=1)
    return np.mean(R),np.std(R)*2/np.sqrt(len(R)),np.mean(opt-R),np.std(opt-R)*2/np.sqrt(len(R))

def stats_terms(A,R,Pi,B,Z=None):
    N = A.shape[0]
    T = A.shape[1]
    if Z is None:Z = np.ones([N,T,1])
    X = np.dstack([B,(A-Pi)[:,:,np.newaxis]*Z])
    dn = X.shape[-1]
    d2 = Z.shape[-1]
    term1 = np.zeros([dn,dn])
    term2 = np.zeros(dn)
    term3 = np.zeros([dn,dn])

    # hat_theta
    for n in range(N):
        idx = np.nonzero(Pi[n]*(1-Pi[n]))        
        denom = (Pi[n]*(1-Pi[n]))[idx]
        Xtmp = X[n][idx]

        term1 += (np.sum(np.array([np.matmul(x[:,np.newaxis],x[:,np.newaxis].T) for x in Xtmp])/denom[:,np.newaxis,np.newaxis],axis=0))/N
        term2 += (np.sum(Xtmp*R[n][idx][:,np.newaxis]/denom[:,np.newaxis],axis=0))/N

    L = np.linalg.inv(np.linalg.cholesky(term1))
    term1_inv = np.matmul(L.T,L)
    theta = np.matmul(term1_inv, term2)
    # cov_hat
    for n in range(N):
        idx = np.nonzero(Pi[n]*(1-Pi[n])) 
        denom = (Pi[n]*(1-Pi[n]))[idx][:,np.newaxis]
        Xtmp = X[n][idx]
        res = R[n][idx]-np.dot(Xtmp,theta)
        res = np.sum(res[:,np.newaxis]*Xtmp/denom,axis=0,keepdims=True)
        term3 += np.matmul(res.T,res)/N
    

    # statistics = N*delta*cov_{-1}*delta
    delta = theta[-d2:]
    return delta, term1, term3

def get_stats(deltas,term1,term2,N):
    d = deltas.shape[1]
    L = np.linalg.inv(np.linalg.cholesky(term2))
    term2_inv = np.matmul(L.T,L)
    cov_inv = np.matmul(term1,np.matmul(term2_inv,term1))[-d:,-d:]
    stats = N*np.squeeze(np.matmul(np.expand_dims(deltas,axis=1),np.matmul(cov_inv,np.expand_dims(deltas,axis=2))))
    return stats

