'''
美式期权可以用过BAW求出近似解
'''

import numpy as np
import scipy.stats as stats
from numpy import sqrt,exp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numba
import math
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
ITERATION_MAX_ERROR = 0.00001 #牛顿迭代法的精度

def BSM(CP,S,X,sigma,T,r,b):
    """
    Parameters
    ----------
    CP：看涨或看跌"C"or"P"
    S : 标的价格.
    X : 行权价格.
    sigma :波动率.
    T : 年化到期时间.
    r : 收益率.
    b : 持有成本，当b = r 时，为标准的无股利模型，b=0时，为black76，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
    Returns
    -------
    返回欧式期权的估值
    """
    d1 = (np.log(S/X) + (b + sigma**2/2)*T) / (sigma* sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if CP == "C":
        value = S * exp((b - r)*T) *stats.norm.cdf(d1) - X * exp(-r*T) * stats.norm.cdf(d2)
    else:
        value =  X * exp(-r*T) * stats.norm.cdf(-d2) - S * exp((b - r)*T) *stats.norm.cdf(-d1)
    return value


def find_Sx(CP,X,sigma,T,r,b):  #手动写的标准的牛顿迭代法
    M = 2 * r / sigma**2
    N = 2 * b /sigma**2
    K = 1 -exp(-r*T)
    q1 = (-(N-1) - sqrt((N-1)**2 + 4*M/K))/2
    q2 = (-(N-1) + sqrt((N-1)**2 + 4*M/K))/2
    if CP == "C":
        S_infinite = X / (1-2*(-(N-1) + sqrt((N-1)**2 + 4*M))**-1)  # 到期时间为无穷时的价格
        h2 = -(b*T + 2*sigma*sqrt(T)) * X / (S_infinite-X)
        Si = X + (S_infinite - X) * (1 - exp(h2))  #计算种子值
        #print(f"Si的种子值为{Si}")
        LHS = Si - X
        d1 = (np.log(Si/X) + (b + sigma**2/2)*T) / (sigma* sqrt(T))
        RHS = BSM("C",Si,X,sigma,T,r,b) + (1 - exp((b-r)*T) * stats.norm.cdf(d1)) * Si / q2
        bi = exp((b-r)*T) * stats.norm.cdf(d1) * (1 - 1/q2)    \
             + (1 - (exp((b-r)*T) * stats.norm.pdf(d1))/ sigma / sqrt(T)) / q2  # bi为迭代使用的初始斜率
        while np.abs((LHS - RHS)/X) > ITERATION_MAX_ERROR:
            Si = (X + RHS  - bi * Si) / (1 - bi)
            #print(f"Si的值迭代为{Si}")
            LHS = Si - X
            d1 = (np.log(Si/X) + (b + sigma**2/2)*T) / (sigma* sqrt(T))
            RHS = BSM("C",Si,X,sigma,T,r,b) + (1 - exp((b-r)*T) * stats.norm.cdf(d1)) * Si / q2
            bi = exp((b-r)*T) * stats.norm.cdf(d1) * (1 - 1/q2)    \
                 + (1 - (exp((b-r)*T) * stats.norm.pdf(d1))/ sigma / sqrt(T)) / q2
        return Si
    else:
        S_infinite= X / (1-2*(-(N-1) - sqrt((N-1)**2 + 4*M))**-1) 
        h1 = -(b*T - 2*sigma*sqrt(T)) * X / (X - S_infinite)
        Si = S_infinite + (X - S_infinite) * exp(h1)  #计算种子值
        #print(f"Si的种子值为{Si}")
        LHS = X - Si
        d1 = (np.log(Si/X) + (b + sigma**2/2)*T) / (sigma* sqrt(T))
        RHS = BSM("P",Si,X,sigma,T,r,b) - (1 - exp((b-r)*T) * stats.norm.cdf(-d1)) * Si / q1
        bi = -exp((b-r)*T) * stats.norm.cdf(-d1) * (1 - 1/q1)    \
             - (1 + (exp((b-r)*T) * stats.norm.pdf(-d1))/ sigma / sqrt(T)) / q1
        while np.abs((LHS - RHS)/X) > ITERATION_MAX_ERROR:
            Si = (X - RHS  + bi * Si) / (1 + bi)
            #print(f"Si的值迭代为{Si}")
            LHS = X - Si
            d1 = (np.log(Si/X) + (b + sigma**2/2)*T) / (sigma* sqrt(T))
            RHS = BSM("P",Si,X,sigma,T,r,b) - (1 - exp((b-r)*T) * stats.norm.cdf(-d1)) * Si / q1
            bi = -exp((b-r)*T) * stats.norm.cdf(-d1) * (1 - 1/q1)    \
                 - (1 + (exp((b-r)*T) * stats.norm.pdf(-d1))/ sigma / sqrt(T)) / q1
        return Si
    
def find_Sx_func(CP,S,X,sigma,T,r,b):
    """
    优化包调用，计算出S^*
    """
    M = 2*r/sigma**2
    N = 2*b/sigma**2
    K = 1-exp(-r*T)
    q1 = (-(N-1)-sqrt((N-1)**2+4*M/K))/2
    
    q2 = (-(N-1)+sqrt((N-1)**2+4*M/K))/2
    d1 = (np.log(S/X) + (b + sigma**2/2)*T) / (sigma* sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    if CP=='C':
        LHS = S-X
        RHS = BSM('C',S,X,sigma,T,r,b) + (1-exp((b-r)*T)*stats.norm.cdf(d1))*S/q2
        Y = (RHS-LSH)**2
    else:
        LHS = X-S
        RHS = BSM('P',S,X,sigma,T,r,b) - (1-exp((b-r)*T)*stats.norm.cdf(-d1))*S/q1
        Y = (RHS-LHS)**2
    return Y

def find_Sx_opt(CP,S,X,sigma,T,r,b):
    """
    利用优化包，计算出S^*
    """
    start =S
    func = lambda S: find_Sx_func(CP,S,X,sigma,T,r,b)
    Ss = opt.fmin(func,start)   
    return Ss

def BAW(CP,S,X,sigma,T,r,b,opt_method="newton"):
    if b > r :#b>r时，美式期权价值和欧式期权相同
        value = BSM(CP,S,X,sigma,T,r,b)

    else:
        M = 2 * r / sigma**2
        N = 2 * b /sigma**2
        K = 1 -exp(-r*T)
        if opt_method=="newton":  #若为牛顿法就用第一种迭代
            Si = find_Sx(CP,X,sigma,T,r,b)
        else: #若不为牛顿法，其他方法这里就是scipy的优化方法
            Si = find_Sx_opt(CP,S,X,sigma,T,r,b)
        d1 = (np.log(Si/X) + (b + sigma**2/2)*T) / (sigma* sqrt(T))
        if CP == "C":   
            q2 = (-(N-1) + sqrt((N-1)**2 + 4*M/K))/2
            A2 = Si/q2 * (1- exp((b-r)*T) * stats.norm.cdf(d1))
            if S < Si:
                value = BSM(CP,S,X,sigma,T,r,b) + A2 * (S/Si)**q2
            else:
                value =  S - X 

        else:
            q1 = (-(N-1) - sqrt((N-1)**2 + 4*M/K))/2
            A1 = - Si/q1 * (1 - exp((b-r)*T) * stats.norm.cdf(-d1))
            if S > Si:
                value = BSM(CP,S,X,sigma,T,r,b) + A1 * (S/Si)**q1
            else:
                value = X- S             
    return value

if __name__ == "__main__":
    a1 = BAW(CP="P",S=100,X=99,sigma=0.2,T=1,r=0.03,b=0,opt_method="scipy")
    a2 = BAW(CP="P",S=100,X=99,sigma=0.2,T=1,r=0.03,b=0,opt_method="newton")
    print(a1)
    print(a2)

