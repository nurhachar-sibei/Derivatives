import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class BSM:
    def __init__(self,CP,S,X,sigma,T,r,b):
        self.CP = CP
        self.S = S
        self.X = X
        self.T = T
        self.r = r
        self.b = b
        self.sigma = sigma

        self.d1 = None
        self.d2 = None
        self.price = self.BSM_Euro_cal(self.CP,self.S,self.X,self.sigma,self.T,self.r,self.b)

        #作图
        #敏感性分析
        self.S_array = self.S
        self.X_array = self.X
        self.T_array = self.T
        self.r_array = self.r
        self.b_array = self.b
        self.sigma_array = self.sigma

        self.V_array_C = None
        self.V_array_P = None

    
    def BSM_Euro_cal(self,CP,S,X,sigma,T,r,b):
        '''
        Parameters
        ----------
        CP : str
            'C' for call option, 'P' for put option
        S : float
            spot price of the underlying asset
        X : float
            strike price of the option
        sigma : float
            volatility of the underlying asset
        T : float
            time to maturity of the option
        r : float
            risk-free interest rate
        b : float
            cost of carry
            持有成本，当b = r 时，为标准的无股利模型，b=0时，为期货期权，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
        Returns
        -------
        float
            fair price of the Euro option
        '''

        self.d1 = (np.log(S/X) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)
        if CP == 'C':
            price = S*np.exp((b-r)*T)*stats.norm.cdf(self.d1) - X*np.exp(-r*T)*stats.norm.cdf(self.d2)
        elif CP == 'P':
            price = X*np.exp(-r*T)*stats.norm.cdf(-self.d2) - S*np.exp((b-r)*T)*stats.norm.cdf(-self.d1)
        else:
            raise ValueError("CP must be 'C' or 'P'")
        return price
    
    def plot_sensitivity(self,x_array,array_name):
        import matplotlib.pyplot as plt
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        if array_name == "S":
            self.S_array = x_array
        elif array_name == "X":
            self.X_array = x_array
        elif array_name == "T":
            self.T_array = x_array
        elif array_name == "r":
            self.r_array = x_array
        elif array_name == "b":
            self.b_array = x_array
        elif array_name == "sigma":
            self.sigma_array = x_array
        else:
            raise ValueError("array_name must be 'S','X','T','r','b','sigma'")

        self.V_array_C = self.BSM_Euro_cal(CP = "C",S = self.S_array,X = self.X_array,sigma = self.sigma_array,T = self.T_array,r = self.r_array,b = self.b_array)
        self.V_array_P = self.BSM_Euro_cal(CP = "P",S = self.S_array,X = self.X_array,sigma = self.sigma_array,T = self.T_array,r = self.r_array,b = self.b_array)
        plt.plot(x_array,self.V_array_C,label="看涨期权")
        plt.plot(x_array,self.V_array_P,label="看跌期权")
        plt.xlabel(array_name)
        plt.ylabel("期权价值")
        plt.legend()

        self.S_array = self.S
        self.X_array = self.X
        self.T_array = self.T
        self.r_array = self.r
        self.b_array = self.b
        self.sigma_array = self.sigma

        return
    #解析解
    def Greeks_analytical_solution(self,CP,S,X,sigma,T,r,b):
        """
        Parameters
        ----------
        CP：看涨或看跌"C"or"P"
        S : 标的价格.
        X : 行权价格.
        sigma :波动率.
        T : 年化到期时间.
        r : 收益率.
        b : 持有成本，当b = r 时，为标准的无股利模型，b=0时，为期货期权，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
        Returns
        -------
        返回欧式期权的估值和希腊字母
        """
        self.d1 = (np.log(S/X) + (b + sigma**2/2)*T) / (sigma*np.sqrt(T))
        self.d2 = self.d1 - sigma*np.sqrt(T)
        if CP == 'C':
            price = self.BSM_Euro_cal(CP,S,X,sigma,T,r,b)
            delta = np.exp((b-r)*T)*stats.norm.cdf(self.d1)
            gamma = np.exp((b-r)*T)*stats.norm.pdf(self.d1) / (S*sigma*np.sqrt(T))
            vega = S*np.exp((b-r)*T)*stats.norm.pdf(self.d1) * np.sqrt(T)
            theta = -S*np.exp((b-r)*T)*stats.norm.pdf(self.d1) * sigma / (2*np.sqrt(T)) - (b-r)*S*np.exp((b-r)*T)*stats.norm.cdf(self.d1) - r*X*np.exp(-r*T)*stats.norm.cdf(self.d2)
            if b!=0:
                rho = X*T*np.exp(-r*T)*stats.norm.cdf(self.d2)
            else:
                rhp = -T*np.exp(-r*T)*(S*stats.norm.cdf(self.d1)-X*stats.norm.cdf(self.d2))
        elif CP == 'P':
            price = self.BSM_Euro_cal(CP,S,X,sigma,T,r,b)
            delta = -np.exp((b-r)*T)*stats.norm.cdf(-self.d1)
            gamma = np.exp((b-r)*T)*stats.norm.pdf(self.d1) / (S*sigma*np.sqrt(T))
            vega = S*np.exp((b-r)*T)*stats.norm.pdf(self.d1) * np.sqrt(T)
            theta = -S*np.exp((b-r)*T)*stats.norm.pdf(self.d1) * sigma / (2*np.sqrt(T)) + (b-r)*S*np.exp((b-r)*T)*stats.norm.cdf(-self.d1) + r*X*np.exp(-r*T)*stats.norm.cdf(-self.d2)
            if b!=0:
                rho = -X * T * exp(-r*T) * norm.cdf(-d2)
            else:
                rho = -T * exp(-r*T) * (X*norm.cdf(-d2) - S*norm.cdf(-d1))
        else:
            raise ValueError("CP must be 'C' or 'P'")
        greeks = {"option_value":price,"delta":delta,"gamma":gamma,"vega":vega,"theta":theta,"rho":rho}
        return greeks
def IV(C0,CP,S,X,T,r,b,vol_est= 0.2):
    '''
    求隐含波动率
    Parameters
    ----------
    C0：期权价值
    CP：看涨或看跌"C"or"P"
    S : 标的价格.
    X : 行权价格.
    T : 年化到期时间.
    r : 收益率.
    b : 持有成本，当b = r 时，为标准的无股利模型，b=0时，为期货期权，b为r-q时，为支付股利模型，b为r-rf时为外汇期权.
    vol_est：预计的初始波动率
    Returns
    -------
    返回看涨期权的隐含波动率。
    '''
    start = 0  #初始波动率下限
    end = 2 #初始波动率上限
    c = 1 # 先给定一个值，让循环运转起来
    while abs(c) >= 0.00001: #迭代差异的精度，根据需要调整
        try:
            val = BSM(CP,S,X,vol_est,T,r,b).price 
        except ZeroDivisionError:
            print("期权的内在价值大于期权的价格，无法收敛出波动率，会触发除0错误")
            break
        if val - C0 > 0: #若计算的期权价值大于实际价值，说明使用的波动率偏大
            end = vol_est
            vol_est = (start + end)/2    
            c = end - vol_est
        else: #若计算的期权价值小于实际价值，说明使用的波动率偏小
            start = vol_est
            vol_est = (start + end)/2
            c = start - vol_est
    return round(vol_est,4)

if __name__ == "__main__":
    bsm = BSM(CP = "C",S = 100,X = 100,sigma = 0.2,T = 1,r = 0.05,b = 0.05)
    print(bsm.price)
    print(IV(bsm.price,bsm.CP,bsm.S,bsm.X,bsm.T,bsm.r,bsm.b))