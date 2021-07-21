from environments import *
from utils import *
from algorithms import *
import ipdb

# fixed pi algorithm
def fixed_pi(bandit,delta,f,Z,alpha,beta,N,T,pi_fix = 0.5,S=1000,approx=False):
    # Initialize bandit
    if isinstance(delta,float):df=1
    else:df = len(delta)
    pi_fix = pi_fix

    ############ fixed pi #######
    deltas = np.zeros([S,df])
    term1 = np.zeros([df+1,df+1])
    term2 = np.zeros([df+1,df+1])
    # stats2 = np.zeros(1000)

    As = []
    Rs = []
    Pis = []
    # Ys = []
    for s in range(S):
        bandit.re_init()
        Pi = np.ones([N,T])*pi_fix
        A = bernoulli.rvs(pi_fix,size=(N,T))
        R = bandit.pull(0,A)
        # primary analysis
        if df==1:B = np.ones([N,T,1])
        else:B = np.expand_dims(np.repeat(f[np.newaxis,:],N,0)+np.repeat(np.dot(Z,delta)[np.newaxis,:],N,0)*Pi,axis=2)
        #B = np.repeat(Z[np.newaxis,:],N,0)
        d,s1,s2 = stats_terms(A,R,Pi,B,Z=Z)
        term1+=(s1/S)
        term2+=(s2/S)
        deltas[s] = d
        Rs.append(R)
        As.append(A)
        Pis.append(Pi)
    # calculate test statistics
    stats = get_stats(deltas,term1,term2,N)
    return stats,Rs,Pis,As

def adaptive(bandit,delta,f,Z,Zalg,sig,alpha,beta,N,T,alg_type,pi,gamma = None,S=1000,wrapper='flip',approx=False,special=False,prior_v=1.0):

    if isinstance(delta,float):df=1
    else:df = len(delta)
    
    # solve for pi
    pi_max = np.copy(pi)
    pi_min = 1-pi_max
    # init
    Rs = []
    Pis = []
    As = [] 

    if alg_type=='linucb' and wrapper=='none': cal_stats = False
    else:cal_stats=True
    
    deltas = np.zeros([S,df])
    if special:
        term1 = np.zeros([df*2,df*2])
        term2 = np.zeros([df*2,df*2])
    elif not approx and df!=1:
        term1 = np.zeros([df*2+1,df*2+1])
        term2 = np.zeros([df*2+1,df*2+1])
    else:
        term1 = np.zeros([df*2+1,df*2+1])
        term2 = np.zeros([df*2+1,df*2+1])
    
    # running siimulations
    for s in range(S):
        A = np.zeros([N,T])
        R = np.zeros([N,T])
        Pi = np.zeros([N,T])
        Y = np.zeros([N,T])
        V = []
        vec = []
        tmp = []
        for n in range(N):
            if alg_type=='bose': alg = BOSE(Zalg.shape[1],T,gamma=gamma)
            elif alg_type=='acts':alg = ActionCenteredThompsonSampling(Zalg.shape[1],sig,prior_v=prior_v)
            elif alg_type == 'linucb': alg = LinearUCB(Zalg.shape[1]+1,gamma=gamma)
            else:
                raise Exception('Invalid algorithm')
            # re-initialize bandit
            bandit.re_init()
            for t in range(T):
                z = Zalg[t]
                b = f[t]
                # get action A and policy fromt the input algorithm
                if alg_type=='linucb':
                    a,pi = alg.get_action(b,np.zeros(len(z)),z)
                else:
                    a,pi = alg.get_action(np.zeros(len(z)),z)
                
                #================== Action Flipping ================
                if wrapper == 'flip':
                    # flip action A with some probability and get action A_2
                    if a==1:  a_2 = bernoulli.rvs(pi_max)
                    else: a_2 = bernoulli.rvs(pi_min)
                    # perform A'
                    r = bandit.pull(t,a_2)
                    # algorithm stores A, R'
                    if alg_type=='linucb': alg.update(a,b,r,z)
                    else: alg.update(a,pi,r,z)
                    # scientist stores A', R', pi'
                    A[n,t] = a_2
                    Pi[n,t] = pi*pi_max+(1-pi)*pi_min
                    R[n,t] = r
                #================== Data Dropping ================
                elif wrapper=='drop': 
                    if pi<=pi_min:
                        a_2 = bernoulli.rvs(pi_min)
                        alg_pi = pi*pi_min/(pi*pi_min+(1-pi)*(1-pi_min))
                        pi = np.copy(pi_min)
                    elif pi>=pi_max: 
                        a_2 = bernoulli.rvs(pi_max)
                        alg_pi = pi*pi_max/(pi*pi_max+(1-pi)*(1-pi_max))
                        pi = np.copy(pi_max)
                    else: 
                        a_2 = np.copy(a)
                        alg_pi = np.copy(pi)
                    r = bandit.pull(t,int(a_2))
                    # update algorithm when actions agree
                    if int(a)==int(a_2):
                        if alg_type=='linucb': alg.update(a,b,r,z)
                        else: alg.update(a,alg_pi,r,z)
                    A[n,t] = a_2
                    Pi[n,t] = pi
                    R[n,t] = r
                #================== Probability Clipping ================
                elif wrapper=='clip' or wrapper=='none': 
                    if pi<=pi_min: pi = np.copy(pi_min)
                    elif pi>=pi_max: pi = np.copy(pi_max)
                    a = bernoulli.rvs(pi)
                    r = bandit.pull(t,a)
                    if alg_type=='linucb': alg.update(a,b,r,z)
                    else: alg.update(a,pi,r,z)
                    # calculation related to statistics
                    A[n,t] = a
                    Pi[n,t] = pi
                    R[n,t] = r

                else: raiseError('Invalid wrapper algorithm')
        # calculate test statistics
        if special:
             B = np.repeat(Z[np.newaxis,:],N,0)
        elif not approx and df!=1:   
            B = np.dstack([np.repeat(f[np.newaxis,:,np.newaxis],N,0),\
                np.repeat(Z[np.newaxis,:],N,0)*Pi[:,:,np.newaxis]])
        else: B = np.dstack([np.ones([N,T,1]),np.repeat(Z[np.newaxis,:],N,0)*Pi[:,:,np.newaxis]])
        Rs.append(R)
        Pis.append(Pi)
        As.append(A)
        if cal_stats:
            d,s1,s2 = stats_terms(A,R,Pi,B,Z=Z)
            term1+=(s1/S)
            term2+=(s2/S)
            deltas[s] = d

    if cal_stats:
        stats1 = get_stats(deltas,term1,term2,N)
        return stats1,Rs,Pis,As

    else: return As,Rs

