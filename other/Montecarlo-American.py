import numpy as np
import matplotlib.pyplot as plt
import numba
import pandas as pd
import scipy.stats as stats
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def geo_brownian(steps,paths,T,S0,b,sigma):
    '''
    生成几何布朗运动
    steps: 时间步数
    paths: 路径数
    T: 时间长度
    S0: 初始价格
    b:  drift项
    sigma: 波动率
    '''
    dt = T/steps
    S_path = np.zeros((steps+1,paths))
    S_path[0] = S0
    for step in range(1,steps+1):
        rn = np.random.standard_normal(paths)
        S_path[step] = S_path[step-1]*np.exp((b-0.5*sigma**2)*dt + sigma*np.sqrt(dt)*rn) #几何布朗运动
    return S_path

def LS_MC(steps,paths,CP,S0,X,sigma,T,r,b):
    '''
    利用LS方法计算蒙特卡洛模拟法下的美式期权价格
    steps: 时间步数
    paths: 路径数
    CP: 期权类型，'C'为看涨期权，'P'为看跌期权
    S0: 初始价格
    X: 行权价格
    sigma: 波动率
    T: 时间长度
    r: 无风险利率
    '''
    #第一步：生成几何布朗运动的价格路径
    S_path = geo_brownian(steps,paths,T,S0,b,sigma)
    dt = T/steps
    cash_flow = np.zeros_like(S_path)#先生成现金流矩阵
    df = np.exp(-r*dt)# discount factor
    #第二步：计算每个时间节点的期权价值
    if CP == 'C':
        cash_flow[-1] = np.maximum(S_path[-1]-X,0)#最后一期的价值
        exercise_value = np.maximum(S_path-X,0)
    elif CP == 'P':
        cash_flow[-1] = np.maximum(X-S_path[-1],0)#最后一期的价值
        exercise_value = np.maximum(X-S_path,0)
    else:
        raise ValueError('CP must be C or P')
    #计算最优决策点：
    for t in range(steps-1,0,-1): #steps-1 ->倒数第二个事件点
        df_cash_flow = cash_flow[t+1]*df #未来一期的现金流折现
        S_price = S_path[t] #当前所有模拟路径下的股价集合
        itm_index = (exercise_value[t] > 0) #当前时间下实指的index，用于后续的回归
        reg = np.polyfit(x=S_price[itm_index],y=df_cash_flow[itm_index],deg=2) # 实值路径下的标的股价X和下一期的折现现金流Y回归
        holding_value = exercise_value[t].copy()
        holding_value[itm_index] = np.polyval(reg,S_price[itm_index])
        ex_index = itm_index &(exercise_value[t] > holding_value) #在实值路径上，进一步寻找出提前行权的index
        df_cash_flow[ex_index] = exercise_value[t][ex_index] #提前行权的路径下，现金流为行权价值
        cash_flow[t] = df_cash_flow
    value = cash_flow[1].mean()*df
    return value

if __name__ == '__main__':
    steps = 1000
    paths = 50000
    CP = 'P'
    S0 = 40
    X = 40
    sigma = 0.2
    T = 1
    r = 0.06
    b = 0.06
    value = LS_MC(steps,paths,CP,S0,X,sigma,T,r,b)
    print(value)


