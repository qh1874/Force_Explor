import arms
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from tracker import Tracker2, SWTracker, DiscountTracker
from MAB import GenericMAB as GMAB
from generate_data import generate_arm_Gauss,generate_arm_Beronoulli
from utils import plot_mean_arms, traj_arms,save_data
from param_stationary import *


T=param['T'] # Number of rounds
K=param['K'] # Number of Arms
m=param['m'] # Length of stationary phase, breakpoints=T/m
N=param['N'] # Repeat Times
  

print("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
seed=1
arm_start,param_start,chg_dist=generate_arm_Beronoulli(T,K,m,seed)
#arm_start,param_start,chg_dist=generate_arm_Gauss(T,K,m,seed)
mab = GMAB(arm_start, param_start, chg_dist)
FE_exp_data=mab.MC_regret("FE_exp", N, T, {}, store_step=1)
FE_lin_data= mab.MC_regret("FE_lin", N, T, {}, store_step=1)
FE_con_data = mab.MC_regret("FE_con", N, T, {}, store_step=1)
Lattimore_data = mab.MC_regret('Lattimore', N, T, {}, store_step=1)
LBSDA_data = mab.MC_regret('LB_SDA', N, T, param_lbsda, store_step=1)
        
x=np.arange(T)
d = int(T / 20)
dd=int(T/1000)
xx = np.arange(0, T, d)
xxx=np.arange(0,T,dd)
alpha=0.05
plt.figure(2)
FE_exp_data1=FE_exp_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(FE_exp_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, FE_exp_data[0][xx], '-r^', markerfacecolor='none', label='FE-Exp')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.25,color='r')

FE_lin_data1=FE_exp_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(FE_lin_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, FE_lin_data[0][xx], '-g*', markerfacecolor='none', label='FE-Linear')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.25,color='g')

FE_con_data1=FE_con_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(FE_con_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, FE_con_data[0][xx], '-ms', markerfacecolor='none', label='FE-Constant')#ms
plt.fill_between(xxx, low_bound, high_bound, alpha=0.25,color='m')

LBSDA_data1=LBSDA_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(LBSDA_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, LBSDA_data[0][xx], '-c*', markerfacecolor='none', label='LB-SDA-LM')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.25,color='c')

Lattimore_data1=Lattimore_data[1].T[:,xxx]
low_bound, high_bound = sms.DescrStatsW(Lattimore_data1).tconfint_mean(alpha=alpha)
plt.plot(xx, Lattimore_data[0][xx], '-k^', markerfacecolor='none', label='Lattimore(2017)')
plt.fill_between(xxx, low_bound, high_bound, alpha=0.25,color='k')

plt.legend()
#plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
plt.xlabel('Round t')
plt.ylabel('Regret')
plt.show()
#plt.savefig('pics/temp/result_exp3.pdf')
plt.close()
