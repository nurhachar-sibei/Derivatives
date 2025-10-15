"""
美式期权定价类
整合了BAW近似解法、二叉树方法和Monte Carlo LSM方法
原BAW与Binarytree放入other文件夹
"""

import numpy as np
import scipy.stats as stats
from numpy import sqrt, exp
import scipy.optimize as opt
import matplotlib.pyplot as plt
import numba
import math
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class AmericanOption:
    """
    美式期权定价类
    
    支持三种定价方法：
    1. BAW近似解法 (Barone-Adesi-Whaley)
    2. 二叉树方法 (Binomial Tree)
    3. Monte Carlo LSM方法 (Longstaff-Schwartz)
    """
    
    def __init__(self, option_type, spot_price, strike_price, volatility, 
                 time_to_expiry, risk_free_rate, cost_of_carry):
        """
        初始化美式期权参数
        
        Parameters:
        -----------
        option_type(CP) : str
            期权类型，"C"为看涨期权，"P"为看跌期权
        spot_price(S) : float
            标的资产当前价格
        strike_price(X) : float
            行权价格
        volatility(sigma) : float
            波动率
        time_to_expiry(T) : float
            年化到期时间
        risk_free_rate(r) : float
            无风险利率
        cost_of_carry(b) : float
            持有成本，当b = r时为标准无股利模型，b=0时为black76，
            b为r-q时为支付股利模型，b为r-rf时为外汇期权
        """
        self.option_type = option_type
        self.spot_price = spot_price
        self.strike_price = strike_price
        self.volatility = volatility
        self.time_to_expiry = time_to_expiry
        self.risk_free_rate = risk_free_rate
        self.cost_of_carry = cost_of_carry
        
        # 迭代精度
        self.ITERATION_MAX_ERROR = 0.00001
    
    def bsm_european_price(self, spot=None, strike=None, vol=None, 
                          time=None, rate=None, carry=None):
        """
        计算欧式期权价格 (Black-Scholes-Merton)
        
        Parameters:
        -----------
        spot, strike, vol, time, rate, carry : float, optional
            如果提供，则使用这些参数；否则使用实例属性
            
        Returns:
        --------
        float : 欧式期权价格
        """
        S = spot if spot is not None else self.spot_price
        X = strike if strike is not None else self.strike_price
        sigma = vol if vol is not None else self.volatility
        T = time if time is not None else self.time_to_expiry
        r = rate if rate is not None else self.risk_free_rate
        b = carry if carry is not None else self.cost_of_carry
        
        d1 = (np.log(S/X) + (b + sigma**2/2)*T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)
        
        if self.option_type == "C":
            value = S * exp((b - r)*T) * stats.norm.cdf(d1) - X * exp(-r*T) * stats.norm.cdf(d2)
        else:
            value = X * exp(-r*T) * stats.norm.cdf(-d2) - S * exp((b - r)*T) * stats.norm.cdf(-d1)
        
        return value
    
    def _find_critical_price_newton(self):
        """
        使用牛顿迭代法计算临界价格S*
        
        Returns:
        --------
        float : 临界价格
        """
        X = self.strike_price
        sigma = self.volatility
        T = self.time_to_expiry
        r = self.risk_free_rate
        b = self.cost_of_carry
        
        M = 2 * r / sigma**2
        N = 2 * b / sigma**2
        K = 1 - exp(-r*T)
        q1 = (-(N-1) - sqrt((N-1)**2 + 4*M/K))/2
        q2 = (-(N-1) + sqrt((N-1)**2 + 4*M/K))/2
        
        if self.option_type == "C":
            S_infinite = X / (1 - 2*(-(N-1) + sqrt((N-1)**2 + 4*M))**-1)
            h2 = -(b*T + 2*sigma*sqrt(T)) * X / (S_infinite - X)
            Si = X + (S_infinite - X) * (1 - exp(h2))
            
            LHS = Si - X
            d1 = (np.log(Si/X) + (b + sigma**2/2)*T) / (sigma * sqrt(T))
            RHS = self.bsm_european_price(spot=Si) + (1 - exp((b-r)*T) * stats.norm.cdf(d1)) * Si / q2
            bi = exp((b-r)*T) * stats.norm.cdf(d1) * (1 - 1/q2) + \
                 (1 - (exp((b-r)*T) * stats.norm.pdf(d1)) / sigma / sqrt(T)) / q2
            
            while np.abs((LHS - RHS)/X) > self.ITERATION_MAX_ERROR:
                Si = (X + RHS - bi * Si) / (1 - bi)
                LHS = Si - X
                d1 = (np.log(Si/X) + (b + sigma**2/2)*T) / (sigma * sqrt(T))
                RHS = self.bsm_european_price(spot=Si) + (1 - exp((b-r)*T) * stats.norm.cdf(d1)) * Si / q2
                bi = exp((b-r)*T) * stats.norm.cdf(d1) * (1 - 1/q2) + \
                     (1 - (exp((b-r)*T) * stats.norm.pdf(d1)) / sigma / sqrt(T)) / q2
            
            return Si
        
        else:  # Put option
            S_infinite = X / (1 - 2*(-(N-1) - sqrt((N-1)**2 + 4*M))**-1)
            h1 = -(b*T - 2*sigma*sqrt(T)) * X / (X - S_infinite)
            Si = S_infinite + (X - S_infinite) * exp(h1)
            
            LHS = X - Si
            d1 = (np.log(Si/X) + (b + sigma**2/2)*T) / (sigma * sqrt(T))
            RHS = self.bsm_european_price(spot=Si) - (1 - exp((b-r)*T) * stats.norm.cdf(-d1)) * Si / q1
            bi = -exp((b-r)*T) * stats.norm.cdf(-d1) * (1 - 1/q1) - \
                 (1 + (exp((b-r)*T) * stats.norm.pdf(-d1)) / sigma / sqrt(T)) / q1
            
            while np.abs((LHS - RHS)/X) > self.ITERATION_MAX_ERROR:
                Si = (X - RHS + bi * Si) / (1 + bi)
                LHS = X - Si
                d1 = (np.log(Si/X) + (b + sigma**2/2)*T) / (sigma * sqrt(T))
                RHS = self.bsm_european_price(spot=Si) - (1 - exp((b-r)*T) * stats.norm.cdf(-d1)) * Si / q1
                bi = -exp((b-r)*T) * stats.norm.cdf(-d1) * (1 - 1/q1) - \
                     (1 + (exp((b-r)*T) * stats.norm.pdf(-d1)) / sigma / sqrt(T)) / q1
            
            return Si
    
    def _critical_price_objective(self, S):
        """
        用于优化求解临界价格的目标函数
        
        Parameters:
        -----------
        S : float
            资产价格
            
        Returns:
        --------
        float : 目标函数值
        """
        X = self.strike_price
        sigma = self.volatility
        T = self.time_to_expiry
        r = self.risk_free_rate
        b = self.cost_of_carry
        
        M = 2*r/sigma**2
        N = 2*b/sigma**2
        K = 1-exp(-r*T)
        q1 = (-(N-1)-sqrt((N-1)**2+4*M/K))/2
        q2 = (-(N-1)+sqrt((N-1)**2+4*M/K))/2
        
        d1 = (np.log(S/X) + (b + sigma**2/2)*T) / (sigma * sqrt(T))
        
        if self.option_type == 'C':
            LHS = S - X
            RHS = self.bsm_european_price(spot=S) + (1-exp((b-r)*T)*stats.norm.cdf(d1))*S/q2
        else:
            LHS = X - S
            RHS = self.bsm_european_price(spot=S) - (1-exp((b-r)*T)*stats.norm.cdf(-d1))*S/q1
        
        return (RHS - LHS)**2
    
    def _find_critical_price_optimization(self):
        """
        使用优化方法计算临界价格S*
        
        Returns:
        --------
        float : 临界价格
        """
        start = self.spot_price
        result = opt.minimize_scalar(self._critical_price_objective, 
                                   bounds=(0.01, 10*self.strike_price), 
                                   method='bounded')
        return result.x
    
    def baw_price(self, optimization_method="newton"):
        """
        使用BAW方法计算美式期权价格
        
        Parameters:
        -----------
        optimization_method : str
            优化方法，"newton"使用牛顿迭代法，其他使用scipy优化
            
        Returns:
        --------
        float : 美式期权价格
        """
        # 当b > r时，美式期权价值等于欧式期权价值
        if self.cost_of_carry > self.risk_free_rate:
            return self.bsm_european_price()
        
        M = 2 * self.risk_free_rate / self.volatility**2
        N = 2 * self.cost_of_carry / self.volatility**2
        K = 1 - exp(-self.risk_free_rate * self.time_to_expiry)
        
        # 计算临界价格
        if optimization_method == "newton":
            Si = self._find_critical_price_newton()
        else:
            Si = self._find_critical_price_optimization()
        
        d1 = (np.log(Si/self.strike_price) + (self.cost_of_carry + self.volatility**2/2)*self.time_to_expiry) / \
             (self.volatility * sqrt(self.time_to_expiry))
        
        if self.option_type == "C":
            q2 = (-(N-1) + sqrt((N-1)**2 + 4*M/K))/2
            A2 = Si/q2 * (1 - exp((self.cost_of_carry - self.risk_free_rate)*self.time_to_expiry) * stats.norm.cdf(d1))
            
            if self.spot_price < Si:
                value = self.bsm_european_price() + A2 * (self.spot_price/Si)**q2
            else:
                value = self.spot_price - self.strike_price
        
        else:  # Put option
            q1 = (-(N-1) - sqrt((N-1)**2 + 4*M/K))/2
            A1 = -Si/q1 * (1 - exp((self.cost_of_carry - self.risk_free_rate)*self.time_to_expiry) * stats.norm.cdf(-d1))
            
            if self.spot_price > Si:
                value = self.bsm_european_price() + A1 * (self.spot_price/Si)**q1
            else:
                value = self.strike_price - self.spot_price
        
        return value
    
    def binomial_tree_price(self, steps=1000):
        """
        使用二叉树方法计算美式期权价格
        
        Parameters:
        -----------
        steps : int
            二叉树的步数，默认1000
            
        Returns:
        --------
        float : 美式期权价格
        """
        dt = self.time_to_expiry / steps
        u = math.exp(self.volatility * math.sqrt(dt))
        d = 1 / u
        p = (math.exp(self.cost_of_carry * dt) - d) / (u - d)
        
        # 构建价格树
        S = np.zeros((steps + 1, steps + 1))
        S[0, 0] = self.spot_price
        
        for i in range(1, steps + 1):
            for j in range(i):
                S[j, i] = S[j, i-1] * u
                S[j+1, i] = S[j, i-1] * d
        
        # 计算内在价值
        if self.option_type == 'C':
            intrinsic_value = np.maximum(S - self.strike_price, 0)
        else:
            intrinsic_value = np.maximum(self.strike_price - S, 0)
        
        # 期权价值树
        option_value = np.zeros_like(S)
        option_value[:, -1] = intrinsic_value[:, -1]
        
        # 向后递推
        for i in range(steps - 1, -1, -1):
            for j in range(i + 1):
                continuation_value = (option_value[j, i+1] * p + option_value[j+1, i+1] * (1-p)) / np.exp(self.risk_free_rate * dt)
                option_value[j, i] = max(continuation_value, intrinsic_value[j, i])
        
        return option_value[0, 0]
    
    def _generate_geometric_brownian_motion(self, steps, paths):
        """
        生成几何布朗运动路径
        
        Parameters:
        -----------
        steps : int
            时间步数
        paths : int
            路径数
            
        Returns:
        --------
        np.ndarray : 价格路径矩阵，形状为(steps+1, paths)
        """
        dt = self.time_to_expiry / steps
        S_path = np.zeros((steps + 1, paths))
        S_path[0] = self.spot_price
        
        for step in range(1, steps + 1):
            rn = np.random.standard_normal(paths)
            S_path[step] = S_path[step-1] * np.exp(
                (self.cost_of_carry - 0.5 * self.volatility**2) * dt + 
                self.volatility * np.sqrt(dt) * rn
            )
        
        return S_path
    
    def monte_carlo_lsm_price(self, steps=1000, paths=50000, random_seed=None):
        """
        使用Monte Carlo LSM方法计算美式期权价格
        
        Parameters:
        -----------
        steps : int
            时间步数，默认1000
        paths : int
            模拟路径数，默认50000
        random_seed : int, optional
            随机种子，用于结果重现
            
        Returns:
        --------
        float : 美式期权价格
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # 第一步：生成几何布朗运动的价格路径
        S_path = self._generate_geometric_brownian_motion(steps, paths)
        dt = self.time_to_expiry / steps
        cash_flow = np.zeros_like(S_path)  # 现金流矩阵
        df = np.exp(-self.risk_free_rate * dt)  # 折现因子
        
        # 第二步：计算每个时间节点的期权价值
        if self.option_type == 'C':
            cash_flow[-1] = np.maximum(S_path[-1] - self.strike_price, 0)  # 最后一期的价值
            exercise_value = np.maximum(S_path - self.strike_price, 0)
        elif self.option_type == 'P':
            cash_flow[-1] = np.maximum(self.strike_price - S_path[-1], 0)  # 最后一期的价值
            exercise_value = np.maximum(self.strike_price - S_path, 0)
        else:
            raise ValueError('option_type must be C or P')
        
        # 第三步：计算最优决策点（向后递推）
        for t in range(steps - 1, 0, -1):  # steps-1 -> 倒数第二个时间点
            df_cash_flow = cash_flow[t + 1] * df  # 未来一期的现金流折现
            S_price = S_path[t]  # 当前所有模拟路径下的股价集合
            itm_index = (exercise_value[t] > 0)  # 当前时间下实值的index，用于后续的回归
            
            if np.sum(itm_index) > 0:  # 确保有实值期权
                # 实值路径下的标的股价X和下一期的折现现金流Y回归
                reg = np.polyfit(x=S_price[itm_index], y=df_cash_flow[itm_index], deg=2)
                holding_value = exercise_value[t].copy()
                holding_value[itm_index] = np.polyval(reg, S_price[itm_index])
                
                # 在实值路径上，进一步寻找出提前行权的index
                ex_index = itm_index & (exercise_value[t] > holding_value)
                # 提前行权的路径下，现金流为行权价值
                df_cash_flow[ex_index] = exercise_value[t][ex_index]
            
            cash_flow[t] = df_cash_flow
        
        # 第四步：计算期权价值
        value = cash_flow[1].mean() * df
        return value
    
    def price(self, method="baw", **kwargs):
        """
        计算美式期权价格的统一接口
        
        Parameters:
        -----------
        method : str
            定价方法，"baw"使用BAW方法，"binomial"使用二叉树方法，"monte_carlo"使用Monte Carlo LSM方法
        **kwargs : dict
            其他参数，传递给具体的定价方法
            
        Returns:
        --------
        float : 美式期权价格
        """
        if method.lower() == "baw":
            return self.baw_price(**kwargs)
        elif method.lower() == "binomial":
            return self.binomial_tree_price(**kwargs)
        elif method.lower() in ["monte_carlo", "montecarlo", "lsm"]:
            return self.monte_carlo_lsm_price(**kwargs)
        else:
            raise ValueError("方法必须是 'baw', 'binomial' 或 'monte_carlo'")
    
    def compare_methods(self, time_array=None, plot=True, mc_params=None):
        """
        比较不同定价方法的结果
        
        Parameters:
        -----------
        time_array : array-like, optional
            时间数组，默认为0.01到2年的50个点
        plot : bool
            是否绘制比较图，默认True
        mc_params : dict, optional
            Monte Carlo方法的参数，包含steps, paths, random_seed
            
        Returns:
        --------
        dict : 包含各方法价格的字典
        """
        if time_array is None:
            time_array = np.linspace(0.01, 2, 50)
        
        if mc_params is None:
            mc_params = {"steps": 500, "paths": 10000, "random_seed": 42}
        
        baw_prices = []
        binomial_prices = []
        monte_carlo_prices = []
        european_prices = []
        
        original_time = self.time_to_expiry
        
        for T in time_array:
            self.time_to_expiry = T
            baw_prices.append(self.baw_price())
            binomial_prices.append(self.binomial_tree_price())
            monte_carlo_prices.append(self.monte_carlo_lsm_price(**mc_params))
            european_prices.append(self.bsm_european_price())
        
        # 恢复原始到期时间
        self.time_to_expiry = original_time
        
        results = {
            'time': time_array,
            'baw': np.array(baw_prices),
            'binomial': np.array(binomial_prices),
            'monte_carlo': np.array(monte_carlo_prices),
            'european': np.array(european_prices)
        }
        
        if plot:
            plt.figure(figsize=(12, 8))
            plt.plot(time_array, baw_prices, label="BAW美式期权", linewidth=2)
            plt.plot(time_array, binomial_prices, label="二叉树美式期权", linewidth=2, linestyle='--')
            plt.plot(time_array, monte_carlo_prices, label="Monte Carlo LSM美式期权", linewidth=2, linestyle='-.')
            plt.plot(time_array, european_prices, label="BSM欧式期权", linewidth=2, linestyle=':')
            plt.xlabel("到期时间 (年)")
            plt.ylabel("期权价格")
            plt.title("美式期权定价方法比较")
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
        
        return results
    
    def __str__(self):
        """返回期权的字符串表示"""
        option_name = "看涨期权" if self.option_type == "C" else "看跌期权"
        return f"""
美式{option_name}:
  标的价格: {self.spot_price}
  行权价格: {self.strike_price}
  波动率: {self.volatility:.2%}
  到期时间: {self.time_to_expiry:.4f}年
  无风险利率: {self.risk_free_rate:.2%}
  持有成本: {self.cost_of_carry:.2%}
        """


if __name__ == "__main__":
    # 示例用法
    option = AmericanOption(
        option_type="P",
        spot_price=100,
        strike_price=99,
        volatility=0.2,
        time_to_expiry=1,
        risk_free_rate=0.03,
        cost_of_carry=0
    )
    
    print(option)
    print(f"BAW价格: {option.baw_price():.4f}")
    print(f"二叉树价格: {option.binomial_tree_price():.4f}")
    print(f"Monte Carlo LSM价格: {option.monte_carlo_lsm_price(steps=1000, paths=50000, random_seed=42):.4f}")
    print(f"欧式期权价格: {option.bsm_european_price():.4f}")
    
    # 比较不同方法
    # option.compare_methods(time_array = np.linspace(0.01,100,50))