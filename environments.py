import numpy as np
from scipy.stats import norm,t,gamma,ncx2,chi2,bernoulli
import ipdb


############## Environments for simulations ############




# Contextual Bandit
class ContextualBandit(object):
    def __init__(self,g,Z,delta,sig=0.5):

        self.g = g
        self.Z = Z
        self.delta = delta
        self.df = len(delta)

        self.sig = sig
    
    def re_init(self):
        pass

    def pull(self,curr_t,a,size=1):
        if isinstance(a,int):
            # select action
            # get reward
            r = a*np.sum(self.Z[curr_t]*self.delta)+self.g[curr_t]+np.random.randn()*self.sig
            return r
        else:
            # select action
            r = a*np.repeat(np.dot(self.Z,self.delta)[np.newaxis,:],a.shape[0],0)+\
                    np.repeat(self.g[np.newaxis,:],a.shape[0],0)+\
                    np.random.randn(a.shape[0],a.shape[1])*self.sig
            return r

# Mobile Health Simulator
class HeartStep(object):
    def __init__(self,gamma,Z,delta,sig=30):

        self.gamma = gamma
        self.Z = Z
        self.delta = delta
        self.df = len(delta)

        self.sig = sig
        self.pre_eps = np.random.randn()*np.sqrt(2)

        
    def re_init(self):
        self.pre_eps = np.random.randn()*np.sqrt(2)
        
        

    def pull(self,t_,a,size=1):
        if isinstance(a,int):
            # get eps
            eps = self.pre_eps/np.sqrt(2)+np.random.randn()
            # get reward
            r = a*np.sum(self.Z[t_]*self.delta)+self.gamma[t_]+eps*self.sig[t_]/np.sqrt(2)
            self.pre_eps = eps
            return r
        else:   
            T = a.shape[1]
            N = a.shape[0]
            e = np.random.randn(N,T)
            # noise follows AR(1) process
            eps = np.zeros([N,T+1])
            eps[:,0] = np.random.randn(N)*np.sqrt(2)
            for i in range(T):
                eps[:,i+1] += (eps[:,i]/np.sqrt(2)+e[:,i])
            eps = eps[:,1:]
            r = a*np.repeat(np.dot(self.Z,self.delta)[np.newaxis,:],N,0)+\
                    np.repeat(self.gamma[np.newaxis,:],N,0)+eps*self.sig/np.sqrt(2)
            return r


