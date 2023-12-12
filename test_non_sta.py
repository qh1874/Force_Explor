import arms
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.stats.api as sms
from tracker import Tracker2, SWTracker, DiscountTracker
from MAB import GenericMAB as GMAB
from generate_data import generate_arm_Gauss,generate_arm_Exp,generate_arm_Pareto,generate_arm_Beronoulli,generate_arm_GB
from utils import plot_mean_arms, traj_arms,save_data
from param import *


T=param['T'] # Number of rounds
K=param['K'] # Number of Arms
m=param['m'] # Length of stationary phase, breakpoints=T/m
N=param['N'] # Repeat Times
  
for k in range(1):  
    # Keep the distribution of arms consistent each run
    seed=0
    arm_start,param_start,chg_dist=generate_arm_Beronoulli(T,K,m,seed)
   
    mab = GMAB(arm_start, param_start, chg_dist)
    FE_data=mab.MC_regret("FE_exp", N, T, {}, store_step=1)
    FE_EXP_data = mab.MC_regret("FE_EXP_SW", N, T, param_fe_exp, store_step=1)
    FE_LIN_data = mab.MC_regret("FE_LIN_SW", N, T, param_fe_lin, store_step=1)
    FE_CON_data = mab.MC_regret("FE_CON_SW", N, T, param_fe_lin, store_step=1)
    SW_TS_data = mab.MC_regret('SW_TS', N, T, param_swts,store_step=1)
    LBSDA_data = mab.MC_regret('LB_SDA', N, T, param_lbsda, store_step=1)
    CUSUM_data = mab.MC_regret('CUSUM', N, T, param_cumsum,store_step=1)
    M_UCB_data = mab.MC_regret('M_UCB', N, T, param_mucb,store_step=1)

    rr=np.zeros(5)
    L=['FE_data','FE_EXP_data','SW_TS_data','LBSDA_data']
    #L=['SW_UCB_data','CUSUM_data','DS_UCB_data','SW_TS_data','M_UCB_data','DS_TS_data']
    ii=0
    for i in L:
        print(i+":",eval(i)[0][-1])
        rr[ii]=eval(i)[0][-1]
        ii += 1
    # #np.save("g_a50b20.npy",rr)
    print("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
   
    x=np.arange(T)
    d = int(T / 20)
    dd=int(T/1000)
    xx = np.arange(0, T, d)
    xxx=np.arange(0,T,dd)
    alpha=0.05
    plt.figure(2)
    FE_EXP_data1=FE_EXP_data[1].T[:,xxx]
    low_bound, high_bound = sms.DescrStatsW(FE_EXP_data1).tconfint_mean(alpha=alpha)
    plt.plot(xx, FE_EXP_data[0][xx], '-r^', markerfacecolor='none', label='SW-FE-Exp')
    plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='r')

    FE_LIN_data1=FE_LIN_data[1].T[:,xxx]
    low_bound, high_bound = sms.DescrStatsW(FE_LIN_data1).tconfint_mean(alpha=alpha)
    plt.plot(xx, FE_LIN_data[0][xx], '-g*', markerfacecolor='none', label='SW-FE-Linear')
    plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='g')
    
    FE_data1=FE_data[1].T[:,xxx]
    low_bound, high_bound = sms.DescrStatsW(FE_data1).tconfint_mean(alpha=alpha)
    plt.plot(xx, FE_data[0][xx], color='brown',marker='*', markerfacecolor='none', label='FE-Exp')
    plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='brown')
    
    LBSDA_data1=LBSDA_data[1].T[:,xxx]
    low_bound, high_bound = sms.DescrStatsW(LBSDA_data1).tconfint_mean(alpha=alpha)
    plt.plot(xx, LBSDA_data[0][xx], '-c*', markerfacecolor='none', label='SW-LB-SDA')
    plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='c')

    FE_CON_data1=FE_CON_data[1].T[:,xxx]
    low_bound, high_bound = sms.DescrStatsW(FE_CON_data1).tconfint_mean(alpha=alpha)
    plt.plot(xx, FE_CON_data[0][xx], '-bd', markerfacecolor='none', label='SW-FE-Constant')
    plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='b')

    M_UCB_data1=M_UCB_data[1].T[:,xxx]
    low_bound, high_bound = sms.DescrStatsW(M_UCB_data1).tconfint_mean(alpha=alpha)
    plt.plot(xx, M_UCB_data[0][xx], '-ms', markerfacecolor='none', label='M-UCB')
    plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='m')

    CUSUM_data1=CUSUM_data[1].T[:,xxx]
    low_bound, high_bound = sms.DescrStatsW(CUSUM_data1).tconfint_mean(alpha=alpha)
    plt.plot(xx, CUSUM_data[0][xx], '-k^', markerfacecolor='none', label='CUSUM')
    plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='k')
    

    SW_TS_data1=SW_TS_data[1].T[:,xxx]
    low_bound, high_bound = sms.DescrStatsW(SW_TS_data1).tconfint_mean(alpha=alpha)
    plt.plot(xx, SW_TS_data[0][xx], '-y^', markerfacecolor='none', label='SW-TS')
    plt.fill_between(xxx, low_bound, high_bound, alpha=0.5,color='y')


    plt.legend()
    #plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    plt.xlabel('Round t')
    plt.ylabel('Regret')
    plt.show()
    #plt.savefig('pics/temp/result'+str(k)+'.png')
    plt.close()
