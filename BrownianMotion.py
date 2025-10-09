import numpy as np
import matplotlib.pyplot as plt
import numba

def standar_brownian(steps,paths,T,S0):
    dt = T / steps # 求出dt
    S_path = np.zeros((steps+1,paths))   #创建一个矩阵，用来准备储存模拟情况
    S_path[0] = S0  #起点设置
    rn = np.random.standard_normal(S_path.shape) # 一次性创建出需要的正态分布随机数，当然也可以写在循环里每次创建一个时刻的随机数
    for step in range(1,steps+1):
        S_path[step] = S_path[step - 1] + rn[step-1]*np.sqrt(dt)
    plt.plot(S_path[:,:])
    plt.show()
    return S_path

def brownian(steps,paths,T,S0,a,b):
    dt = T / steps # 求出dt
    S_path = np.zeros((steps+1,paths))   #创建一个矩阵，用来准备储存模拟情况
    S_path[0] = S0  #起点设置
    rn = np.random.standard_normal(S_path.shape) # 一次性创建出需要的正态分布随机数，当然也可以写在循环里每次创建一个时刻的随机数
    for step in range(1,steps+1):
        S_path[step] = S_path[step - 1] + a*dt + b* rn[step-1]*np.sqrt(dt) # 和标准布朗运动的区别就在这一行
    plt.plot(S_path[:,:])
    plt.show()
    return S_path

def geo_brownian(steps,paths,T,S0,u,sigma):
    dt = T / steps # 求出dt
    S_path = np.zeros((steps+1,paths))   #创建一个矩阵，用来准备储存模拟情况
    S_path[0] = S0  #起点设置
    rn = np.random.standard_normal(S_path.shape) # 一次性创建出需要的正态分布随机数，当然也可以写在循环里每次创建一个时刻的随机数
    for step in range(1,steps+1):
        S_path[step] = S_path[step - 1] * np.exp((u-0.5*sigma**2)*dt +sigma*np.sqrt(dt)*rn[step]) # 和其他布朗运动的区别就在这一行
    plt.plot(S_path[:,:])
    plt.show()
    return S_path

if __name__ == '__main__':
    S_path = geo_brownian(steps = 100,paths = 50,T = 1,S0 = 100,u = 0.03, sigma = 0.2)

