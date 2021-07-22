import sys
import numpy as np
from wrapper_algs import *
from utils import *
from environments import *
# import pickle
import argparse

# argument parser
parser = argparse.ArgumentParser('argment parser')
parser.add_argument('--S',type=int,default=1000) # number of total simulations
parser.add_argument('--env',type=str,default='scb', choices = ['scb', 'ascb', 'mobile', 'nonlinear']) # environments (options: scb, ascb, mobile, nonlinear) 
parser.add_argument('--alg',type=str,default='fix_pi', choices = ['fix_pi', 'acts', 'bose', 'linucb']) # algorithms (options: fix_pi, acts, bose, linucb)
parser.add_argument('--wrapper',type=str,default='clip', choices = ['none', 'clip', 'drop', 'flip']) # wrapper algorithms (options: none, clip, drop, flip)
parser.add_argument('--experiment',type=str,default='power',choices = ['power','type_i']) # experiment type (options: power, type_i)

parser.add_argument('--drop',dest='drop', action='store_true', default=False) # if drop last dimension of treatment effect feature vector Z_nt
parser.add_argument('--approx',dest='approx', action='store_true', default=False) # if approximate marginal reward model as B_nt=1
parser.add_argument('--noise_size',type=str,default='none', choices =['small','none','large']) # estimated noise size (options: none, small, large)
parser.add_argument('--effect_size',type=str,default='none', choices =['small','none','large']) # estimated treatment effect size (options: none, small, large)

args = parser.parse_args()
S = args.S
wrapper = args.wrapper
alg = args.alg
env = args.env  
experiment = args.experiment

noise_effect = args.noise_size
effect = args.effect_size
drop = args.drop
approx = args.approx

# desired type i error = 0.05, desired power = 1-\beta = 0.8
alpha = 0.05; beta = 0.2;
# For each simulation, we collect data from 20 users for 90 steps
N = 20; T = 90; 

# result saving path
exp_des = '{}_{}_alg_{}_noise_{}_effect_{}_approx_{}_drop_{}_{}'.format(env,alg,wrapper,noise_effect,effect,str(approx).lower(),str(drop).lower(),experiment)


# Define environments
if env in ['ascb' ,'scb','nonlinear']: environment = ContextualBandit
else: environment = HeartStep
### ========================= Define Reward Model ========================= ###
####################    ASCB         #############################
if env=='ascb':
    Z = np.array([[-0.5,0.3*(-1)**t,(t/100)**2] for t in range(T)]) # Z_nt
    delta =np.ones(3)*0.2 # delta
    sig = 0.5 # reward model std
    f = -np.max(np.vstack([np.dot(Z,delta),np.zeros(T)]),axis=0) 

####################    SCB         #############################
elif env=='scb':
    D = np.load('scb.npy',allow_pickle=True) # load in Z_nt, delta of SCB enviroment
    Z = D[0] # Z_nt
    delta = D[1] # delta
    sig = 0.5 # reward model std
    f = np.linspace(-0.05,0.05,T)

####################    Mobile Health Simulator         #############################
elif env=='mobile':
    Z = np.array([[1,(t-1)/45,np.square((t-1)/45)] for t in range(T)])  # Z_nt
    x2=-6/((89/45)**2*-9/8+89/45)
    x1=-x2*9/8
    delta =np.array([6,x2,x1])  # delta
    if noise_effect=='none':sig = np.ones(T)*30  # noise std
    else:
        # weekend has higher stochasticity
        sig = np.ones(T)*30
        for i in range(T//7):
            sig[(i+1)*7-1:(i+1)*7+1]=45
    f = np.linspace(125,50,T)
# the following environment is defined for robustness test where the treatment effect model of ascb is mis-specified
# (approximate nonlinear model with a linear model)
else:
    sig = 0.5
    sig_guess = 0.5
    Z = np.array([[-0.5,0.3*(-1)**t,(t/100)**2] for t in range(T)])
    delta =np.ones(3)*0.2

    # define mis-specified treatment effect model
    Z_guess = np.array([[1,t] for t in range(T)])
    delta_guess = np.array([-0.15,0.003])

    f = -np.max(np.vstack([np.dot(Z,delta),np.zeros(T)]),axis=0)
 
    # define estimated treatment effect
    noise_effect='none'; effect='none'; approx = False; drop = False;

# define estimated treatment effect
if env!='nonlinear':
    if effect== 'large':
        delta_guess = delta*1.1 # delta_est > true delta
        Z_guess = np.copy(Z)
    elif effect=='small':
        delta_guess = delta/1.1 # delta_est < true delta
        Z_guess = np.copy(Z)
    else:
        Z_guess = np.copy(Z) # delta_est = true delta
        delta_guess = np.copy(delta)
# define estimated treatment effect
if env in ['ascb','scb']:
    if noise_effect=='none':
        sig_guess = np.copy(sig)
    elif noise_effect=='small':
        sig_guess = sig/np.sqrt(1.2)
    else:
        sig_guess = sig*np.sqrt(1.2)
elif env=='mobile': 
    # defined estimated noise variance for mobile health simulator
    sig_guess = np.sqrt(np.sum(sig**2)/T)

# if test robustness of treatment effect model mis-specification, drop last dimension of the treatment effect feature
if drop:
    Z_guess = Z[:,:-1]
    delta_guess = delta[:-1]

df = len(delta_guess) # degree of freedom
### ========================= Initialize Environment ========================= ###
if experiment=='type_i':
    bandit = environment(f,Z,np.zeros(df),sig=sig) # if type i, treatment effect size = 0
else:
    bandit = environment(f,Z,delta,sig=sig)
exp_des = '{}_{}_alg_{}_noise_{}_effect_{}_approx_{}_drop_{}_{}'.format(env,alg,wrapper,noise_effect,effect,str(approx).lower(),str(drop).lower(),experiment)

print ('Running experiments on {} enviroments with {} and {} wrapper algorithm.'.format(env.upper(),alg.upper(),wrapper))
print ('Robustness setting: estimated treatment effect size {}; estimated noise size {}; mis-specified marginal reward model {};\
        \nmis-specified treatment effect model {};'.format(effect, noise_effect, approx, drop))

### ========================= solve for \pi_min, \pi_max ========================= ###
treat_effect = np.sum(delta*Z,axis=1)
_, clip_pi = solve_pi(sig_guess,delta_guess,alpha,beta,N,T,Z_guess,df = df)


if wrapper == 'none': clip_pi = 1.0
if alg=='bose' and wrapper=='none':special=True
else: special=False
print ('Solved pi_min = {:.3f} and pi_max = {:.3f}'.format(clip_pi,1-clip_pi))

### ========================= Define Hyperparameter for Different Algorithms ========================= ###
hyper_parameters = {'scb_acts':0.15, 'ascb_acts':0.05, 'mobile_acts':60,\
                    'scb_bose':0.2, 'ascb_bose':0.2, 'mobile_bose':120,\
                    'scb_linucb':0.03, 'ascb_linucb':0.02, 'mobile_linucb':95,\
                    'nonlinear_acts':0.05, 'nonlinear_bose':0.2,'nonlinear_linucb':0.02}

### ========================= Running Simulations ========================= ###
if alg=='fix_pi':
    # run algorithm of fixed \pi=0.5
    result = fixed_pi(bandit,delta_guess,f,Z_guess,alpha,beta,N,T,pi_fix = 0.5,S=S,approx=approx)
else:
    k = env+'_'+alg
    if alg =='acts':prior_v = hyper_parameters[k];gamma=None;
    else: gamma = hyper_parameters[k];prior_v=None;

    if env=='mobile': sig = sig[0]
    result = adaptive(bandit,delta_guess,f,Z_guess,Z,sig,alpha,beta,N,T,alg,clip_pi,gamma=gamma,prior_v = prior_v,\
                        special=special,S=S,wrapper=wrapper,approx=approx)


### ========================= Parsing Results ========================= ###
print ('Finished Simulation ....')
if experiment=='type_i':
    e = type_i_error(result[0],df=df)
    print ('\t Resulting type 1 error {:.3f} +/- {:.3f}'.format(e[0],e[1]))

else:
    e = power(result[0],df=df)
    # optimal average return for standard oracle
    opt_r = np.sum(treat_effect[np.where(treat_effect>=0)])+np.sum(f)
    # optimal average return for clipped oracle
    clip_opt_r = np.sum(treat_effect[np.where(treat_effect>=0)])*clip_pi+np.sum(f)
    # regret with respect to standard oracle
    regret = regret_cal(opt_r,result[1])
    # regret with respect to clipped oracle
    regret_c = regret_cal(clip_opt_r,result[1])
    print ('\t Resulting power {:.3f} +/- {:.3f}'.format(e[0],e[1]))
    print ('\t Average Return {:.3f} +/- {:.3f}'.format(regret[0],regret[1]))
    if wrapper=='none':print ('\t Regret w.r.t standard oracle {:.3f} +/- {:.3f}'.format(regret[-2],regret[-1]))
    else:print ('\t Regret w.r.t clipped oracle {:.3f} +/- {:.3f}'.format(regret_c[-2],regret_c[-1]))



    
