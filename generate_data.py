import numpy as np
import matplotlib.pyplot as plt
#plt.rc('text', usetex=True)

def get_reward_distribution_Gauss(T,K,m,seed):
    np.random.seed(seed)
    test_normal=np.zeros((T,K,2))
    xtemp=np.zeros((K,2))
    for ii in range(0,T,m):
       
        xtemp[:,0]=np.random.uniform(0.,1,K)
        xtemp[:,1]=np.random.uniform(0,1,K)
        
        test_normal[ii:ii+m]=xtemp/1
    
    return test_normal

def generate_arm_Gauss(T,K,m,seed):
    test_normal=get_reward_distribution_Gauss(T,K,m,seed)
    KG=['G' for _ in range(K)]
    arm_start=KG
    param_start=test_normal[0].tolist()
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KG,test_normal[i].tolist()]
        
    plt.figure(1)
    for i in range(K):
        plt.plot(test_normal[:,i,0],label='Arm '+str(i+1))
    #plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    plt.legend()
    return arm_start,param_start,chg_dist


def get_reward_distribution_Bernoulli(T,K,m,seed):
    np.random.seed(seed)
    test_bernoulli=np.zeros((T,K))
    xtemp=np.zeros(K)
    for ii in range(0,T,m):
       
        xtemp=np.random.uniform(0,1,K)
        test_bernoulli[ii:ii+m]=xtemp
    
    return test_bernoulli

def generate_arm_Beronoulli(T,K,m,seed):
    test_bernoulli=get_reward_distribution_Bernoulli(T,K,m,seed)
    KB=['B' for _ in range(K)]
    arm_start=KB
    param_start=test_bernoulli[0].tolist()
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KB,test_bernoulli[i].tolist()]
        
    # plt.figure(1)
    # for i in range(K):
    #     plt.plot(test_normal[:,i,0],label='Arm '+str(i+1))
    # #plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    # plt.legend()
    return arm_start,param_start,chg_dist


def generate_arm_GB(T,K,m,seed):
    test_bernoulli=get_reward_distribution_Bernoulli(T,int(K/2),m,seed)
    test_Gauss=get_reward_distribution_Gauss(T,int(K/2),m,seed)
    KB=['B' for _ in range(int(K/2))]
    KG=['G' for _ in range(int(K/2))]
    arm_start=KB + KG
    param_start=test_bernoulli[0].tolist() + test_Gauss[0].tolist()
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KB+KG,test_bernoulli[i].tolist()+test_Gauss[i].tolist()]
        
    # plt.figure(1)
    # for i in range(K):
    #     plt.plot(test_normal[:,i,0],label='Arm '+str(i+1))
    # #plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    # plt.legend()
    return arm_start,param_start,chg_dist

def get_reward_distribution_Exp(T,K,m,seed):
    np.random.seed(seed)
    test_exp=np.zeros((T,K))
    for ii in range(0,T,m):
        test_exp[ii:ii+m]=np.random.uniform(0,10,K)
    
    return test_exp

def generate_arm_Exp(T,K,m,seed):
    test_exp=get_reward_distribution_Exp(T,K,m,seed)
    KExp=['Exp' for _ in range(K)]
    arm_start=KExp
    param_start=test_exp[0].tolist()
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KExp,test_exp[i].tolist()]
        
    # plt.figure(1)
    # for i in range(K):
    #     plt.plot(test_normal[:,i,0],label='Arm '+str(i+1))
    # #plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    # plt.legend()
    return arm_start,param_start,chg_dist
    
    
def get_reward_distribution_Pareto(T,K,m,seed):
    np.random.seed(seed)
    test_pareto=np.zeros((T,K,2))
    xtemp=np.zeros((K,2))
    for ii in range(0,T,m):
        xtemp[:,0]=np.random.uniform(2,10,K)
        xtemp[:,1]=np.random.uniform(1,5,K)
        test_pareto[ii:ii+m]=xtemp#np.random.uniform(0,1,(K,2))
    
    return test_pareto

def generate_arm_Pareto(T,K,m,seed):
    test_pareto=get_reward_distribution_Pareto(T,K,m,seed)
    KP=['Pareto' for _ in range(K)]
    arm_start=KP
    param_start=test_pareto[0].tolist()
    chg_dist={}
    for i in range(m,T,m):
        chg_dist[str(i)]=[KP,test_pareto[i].tolist()]
        
    # plt.figure(1)
    # for i in range(K):
    #     plt.plot(test_normal[:,i,0],label='Arm '+str(i+1))
    # #plt.title("T : {}, arms : {}, breakpoints: {} ".format(T, K, int(T / m)))
    # plt.legend()
    return arm_start,param_start,chg_dist
