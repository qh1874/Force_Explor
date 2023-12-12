import numpy as np

param={
    'T':100000, # round 
    'K':10,  # arm
    'm':100000, # length of stationary phase, breakpoints=T/m
    'N':100 # repeat times
}

T=param['T']
K=param['K']
m=param['m']
nb_change=int(T/m)
Gamma_T_garivier = int(T/m)
reward_u_p = 1
expected_reward_u_p=1
sigma_max = 1
gamma_EXP3 = min(1, 0.1*np.sqrt(K*(nb_change*np.log(K*T)+np.exp(1))/((np.exp(1)-1)*T)))
gamma_D_UCB = 1 # -1/1*np.sqrt(Gamma_T_garivier/T)
gamma_D_UCB_unb = 1# - 1/(4*(reward_u_p+ 2*sigma_max))*np.sqrt(Gamma_T_garivier/T)
tau_theorique = T
tau_no_log = 2*reward_u_p*np.sqrt(T/Gamma_T_garivier)


param_exp3s={'alpha':1/T, 'gamma': gamma_EXP3}
param_dsucb={'B':1,'ksi':8/3, 'gamma': gamma_D_UCB}
#param_swts={'mu_0':0, 'sigma_0':sigma_max, 'sigma':sigma_max, 'tau': int(tau_theorique_unb)}
#param_swts={ 'tau': int(tau_theorique_unb)}
param_dsts={'kexi':1, 'tao_max':expected_reward_u_p/5, 'gamma': 1-10*np.sqrt(nb_change/T)}
param_dsts_b={'gamma': 1-np.sqrt(nb_change/T),'alpha_0':1,'beta_0':1}
param_lbsda={'tau': int(tau_theorique)}
#param_lbsda={}
