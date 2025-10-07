'''
二叉树模型:
用于计算美式期权定价
'''

import pandas as pd
import numpy as np
import math
import numba
def binarytree_am(CP,m,S0,T,sigma,K,r,b):
    '''
    计算美式期权的价格
    参数:
    CP: 期权类型，"C"为看涨期权，"P"为看跌期权
    m: 模拟的期数
    S0: 初始资产价格
    T: 期权到期时间
    sigma: 资产价格的波动率
    K: 期权行权价格
    r: 无风险利率
    b: 持有成本,当b = r 时，为标准的无股利模型，b=0时，为black76，b为r-q时，为支付股利模型，b为r-rf时为外汇期权
    返回:
    美式期权的价格
    '''
    dt = T/m
    u = math.exp(sigma*math.sqrt(dt))
    d = 1/u
    S = np.zeros((m+1,m+1))
    S[0,0] = S0
    p = (math.exp(b*dt) - d)/(u-d)
    for i in range(1,m+1):
        for a in range(i):
            S[a,i] = S[a,i-1] * u
            S[a+1,i] = S[a,i-1] * d
    Sv = np.zeros_like(S)

    if CP == 'C':
        S_intrinsic = np.maximum(S - K,0)
    elif CP == 'P':
        S_intrinsic = np.maximum(K - S,0)
    Sv[:,-1] = S_intrinsic[:,-1]
    for i in range(m-1,-1,-1):
        for a in range(i+1):
            Sv[a,i] = max((Sv[a,i+1] * p + Sv[a+1,i+1] * (1-p))/np.exp(r*dt),S_intrinsic[a,i])
    return Sv[0,0]

if __name__ == '__main__':
    print("start calculate...")
    simulate_tree_nb = numba.jit(binarytree_am) 
    value_option = binarytree_am(CP = 'C',m = 1000,S0 = 100,K=95,T = 1,sigma = 0.25,r = 0.03,b = 0.03)
    print(value_option)

