import numpy as np
from scipy.stats import norm,bernoulli

class ActionCenteredThompsonSampling(object):
    """
    Implementation of Action Centered Thompson Sampling
    """
    def __init__(self,d,sig,prior_v=1.):
        # dimension of Z_t
        self.d = d
        # prior
        self.sig = sig
        self.vec = np.zeros(self.d)
        self.V = np.identity(self.d)*prior_v
        self.V_inv = np.linalg.inv(self.V)
        self.delta = np.matmul(self.V_inv,self.vec)


    def update(self,a,pi,r,z):
        self.V += (np.matmul((a*z[np.newaxis,:]).T,a*z[np.newaxis,:])*pi*(1-pi))
        self.vec+= (a*z*r*(a-pi))
        self.V_inv = np.linalg.inv(self.V)
        self.delta = np.matmul(self.V_inv,self.vec)

    def get_pi(self,z0, z1):
        diff = z1-z0
        if isinstance(diff,float):
            mu = np.squeeze(self.delta*diff)
            var = np.squeeze(self.V_inv*self.sig**2)
        else:
            diff = z1 - z0
            mu = np.dot(diff,self.delta)
            var = np.matmul(diff,np.matmul(self.V_inv*self.sig**2,diff))
        pi = 1-norm.cdf(0,loc=mu,scale=np.sqrt(var))
        return pi


    def get_action(self,z0, z1,misc=True):
        pi = self.get_pi(z0,z1)
        delta_s = np.random.multivariate_normal(self.delta,self.V_inv*self.sig**2)
        a = np.argmax([np.squeeze(np.dot(z0,delta_s)),np.squeeze(np.dot(z1,delta_s))])
        #a = bernoulli.rvs(pi)

        return int(a),pi

class LinearUCB(object):
    """
    Implementation of Linear Upper Confidence Bound
    """
    def __init__(self,d,gamma=1.0,prior_v=1):
        # dimension of Z_t
        self.d = d
        # hyperparameter
        self.gamma = gamma
        # prior
        self.vec = np.zeros(self.d)
        self.V = np.identity(self.d)*prior_v
        self.V_inv = np.linalg.inv(self.V)
        self.theta = np.matmul(self.V_inv,self.vec)

        #clipped probability
        #self.pi_min = pi_min
        #self.pi_max = 1-self.pi_min

    def update(self,a,b,r,z):
        x = np.hstack([b,a*z])
        self.V += np.matmul((x[np.newaxis,:]).T,x[np.newaxis,:])
        self.vec+= (x*r)
        self.V_inv = np.linalg.inv(self.V)
        self.theta = np.matmul(self.V_inv,self.vec)


    def get_pi(self,b,z0,z1):
        x0 = np.hstack([b,z0])
        x1 = np.hstack([b,z1])
        diff = np.dot(x0-x1,self.theta)+self.gamma*(np.sqrt(np.matmul(x0,np.matmul(self.V_inv,x0)))\
            -np.sqrt(np.sqrt(np.matmul(x1,np.matmul(self.V_inv,x1)))))

        if diff<0: 
            return 1.0
            #return self.pi_max
        elif diff>0: 
            return 0.0
            #return self.pi_min
        else:
            return 0.5

    def get_action(self,b,z0, z1,misc=False):
        pi = self.get_pi(b,z0,z1)
        a = bernoulli.rvs(pi)

        return int(a),pi


class BOSE(object):
    """
    Implementation of BOSE (bandit semiparametric orthogonalized
    estimator) algorithm. This algorithm only works if features are
    available in the SemibanditSim object
    The SemibanditSim must also expose B.D as an instance variable
    """
    def __init__(self, d, T, gamma=None):
        self.d = d
        self.T = T
        if gamma is not None:
            self.reg = 1
            self.gamma = gamma
        else:
            self.p = 0.1
            self.reg = 4*self.d*np.log(9*self.T)+8*np.log(4*T/self.p)
            self.gamma = np.sqrt(self.reg)+np.sqrt(27*self.d*np.log(1+2*T/d)+54*np.log(4*T/self.p))
        # init hat_delta,cov
        self.vec = np.zeros(self.d) 
        self.V = np.identity(d)*self.reg
        self.V_inv = np.linalg.inv(self.V)
        self.delta = np.matmul(self.V_inv,self.vec)
        


    def update(self, a, pi,r,z):
        
        mu = pi*z
        self.V += np.matmul((z*a-mu)[:,np.newaxis],(z*a-mu)[np.newaxis,:])
        self.vec += (r*(z*a-mu))

        self.V_inv = np.linalg.inv(self.V)
        self.delta = np.matmul(self.V_inv,self.vec)


    def get_pi(self,z0, z1):
        diff = z1-z0
        ls = np.sqrt(np.matmul(diff,np.matmul(self.V_inv,diff)))*self.gamma
        rs = np.sum(self.delta*diff)
        if rs >= ls:
            pi = 1.0       
            #pi = self.pi_max            
        elif -rs >= ls:
            #pi = self.pi_min 
            pi = 0.0 
        else: pi = 0.5
        return pi

    def get_action(self, z0, z1,misc=False):
        pi = self.get_pi(z0, z1)        
        a = bernoulli.rvs(pi)
        if misc: return a,pi,rs,ls
        else:return a,pi
