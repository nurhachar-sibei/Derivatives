# Black-Scholes-Merton (BSM) 期权定价模型使用说明

## 概述

`BSM.py` 实现了完整的 Black-Scholes-Merton 期权定价模型，提供欧式期权定价、希腊字母计算、敏感性分析和隐含波动率求解等功能。该模型是现代期权定价理论的基础，广泛应用于金融衍生品的估值和风险管理。

## 功能特性

- **欧式期权定价**: 基于BSM公式的精确定价
- **希腊字母计算**: 解析解和数值解两种方法
- **敏感性分析**: 可视化参数对期权价格的影响
- **隐含波动率**: 二分法和牛顿迭代法求解
- **多种市场模型**: 支持无股利、期货、股利、外汇等模型
- **可视化功能**: 内置图表生成功能

## 安装依赖

```bash
pip install numpy scipy matplotlib
```

## 快速开始

### 基本用法

```python
from BSM import BSM

# 创建BSM期权对象
option = BSM(
    CP="C",          # "C"为看涨期权，"P"为看跌期权
    S=100,           # 标的资产当前价格
    X=100,           # 行权价格
    sigma=0.2,       # 波动率 (20%)
    T=1,             # 到期时间 (1年)
    r=0.05,          # 无风险利率 (5%)
    b=0.05           # 持有成本 (5%)
)

# 获取期权价格和希腊字母
print(f"期权价格: {option.price:.4f}")
print(f"希腊字母: {option.greek}")
```

### 隐含波动率计算

```python
from BSM import IV, newton_iv

# 已知期权价格，求隐含波动率
market_price = 10.45
implied_vol_bisection = IV(market_price, "C", 100, 100, 1, 0.05, 0.05)
implied_vol_newton = newton_iv(market_price, "C", 100, 100, 1, 0.05, 0.05)

print(f"隐含波动率 (二分法): {implied_vol_bisection:.4f}")
print(f"隐含波动率 (牛顿法): {implied_vol_newton:.4f}")
```

## 详细API文档

### BSM类

#### 类初始化

```python
BSM(CP, S, X, sigma, T, r, b)
```

**参数说明:**

- `CP` (str): 期权类型

  - `"C"`: 看涨期权 (Call Option)
  - `"P"`: 看跌期权 (Put Option)
- `S` (float): 标的资产当前价格
- `X` (float): 行权价格
- `sigma` (float): 波动率（年化）
- `T` (float): 到期时间（年）
- `r` (float): 无风险利率（年化）
- `b` (float): 持有成本（年化）

  - 当 `b = r` 时：标准无股利模型
  - 当 `b = 0` 时：期货期权模型
  - 当 `b = r - q` 时：支付股利模型（q为股利率）
  - 当 `b = r - rf` 时：外汇期权模型（rf为外币利率）

#### 主要属性

- `price`: 期权价格
- `greek`: 希腊字母（解析解）
- `greek_differential`: 希腊字母（数值解）
- `d1`, `d2`: BSM公式中的参数

#### 主要方法

##### 1. 欧式期权定价

```python
BSM_Euro_cal(CP, S, X, sigma, T, r, b)
```

**返回:** float - 欧式期权价格

##### 2. 敏感性分析图

```python
plot_sensitivity(x_array, array_name)
```

**参数:**

- `x_array`: 参数变化范围的数组
- `array_name`: 参数名称，可选值：`"S"`, `"X"`, `"T"`, `"r"`, `"b"`, `"sigma"`

**功能:** 绘制期权价格对指定参数的敏感性曲线

##### 3. 希腊字母解析解

```python
Greeks_analytical_solution(CP, S, X, sigma, T, r, b)
```

**返回:** dict - 包含所有希腊字母的字典

- `option_value`: 期权价值
- `delta`: 价格敏感性
- `gamma`: Delta的敏感性
- `vega`: 波动率敏感性
- `theta`: 时间衰减
- `rho`: 利率敏感性

##### 4. 希腊字母可视化

```python
plot_greek_sensitivity()
```

**功能:** 绘制希腊字母随标的价格变化的图表

##### 5. 希腊字母数值解

```python
Greeks_differential_solution(CP, S, X, sigma, T, r, b, pct_change=0.0001)
```

**参数:**

- `pct_change`: 数值微分的步长，默认0.01%

**返回:** dict - 希腊字母的数值解

### 独立函数

#### 隐含波动率（二分法）

```python
IV(C0, CP, S, X, T, r, b, vol_est=0.2)
```

**参数:**

- `C0`: 市场期权价格
- 其他参数同BSM类
- `vol_est`: 初始波动率估计

**返回:** float - 隐含波动率

#### 隐含波动率（牛顿迭代法）

```python
newton_iv(C0, CP, S, X, T, r, b, vol_est=0.25, n_iter=1000)
```

**参数:**

- `n_iter`: 最大迭代次数

**返回:** float - 隐含波动率

## 使用示例

### 示例1: 基本期权定价

```python
from BSM import BSM
import numpy as np

# 创建平价看涨期权
call_option = BSM("C", 100, 100, 0.2, 1, 0.05, 0.05)

print("=== 看涨期权信息 ===")
print(f"期权价格: {call_option.price:.4f}")
print(f"Delta: {call_option.greek['delta']:.4f}")
print(f"Gamma: {call_option.greek['gamma']:.4f}")
print(f"Vega: {call_option.greek['vega']:.4f}")
print(f"Theta: {call_option.greek['theta']:.4f}")
print(f"Rho: {call_option.greek['rho']:.4f}")

# 创建对应的看跌期权
put_option = BSM("P", 100, 100, 0.2, 1, 0.05, 0.05)
print(f"\n看跌期权价格: {put_option.price:.4f}")

# 验证看涨看跌平价关系
parity_check = call_option.price - put_option.price - (100 * np.exp(-0.05 * 1) - 100 * np.exp(-0.05 * 1))
print(f"平价关系检验 (应接近0): {parity_check:.6f}")
```

### 示例2: 敏感性分析

```python
import matplotlib.pyplot as plt
import numpy as np

# 创建期权对象
option = BSM("C", 100, 100, 0.2, 1, 0.05, 0.05)

# 分析标的价格敏感性
spot_prices = np.linspace(80, 120, 50)
option.plot_sensitivity(spot_prices, "S")
plt.title("期权价格对标的价格的敏感性")
plt.show()

# 分析波动率敏感性
volatilities = np.linspace(0.1, 0.5, 50)
option.plot_sensitivity(volatilities, "sigma")
plt.title("期权价格对波动率的敏感性")
plt.show()

# 分析时间敏感性
times = np.linspace(0.01, 2, 50)
option.plot_sensitivity(times, "T")
plt.title("期权价格对到期时间的敏感性")
plt.show()
```

### 示例3: 希腊字母分析

```python
# 希腊字母随标的价格变化
option = BSM("C", 100, 100, 0.2, 1, 0.05, 0.05)
option.plot_greek_sensitivity()

# 比较解析解和数值解
print("=== 希腊字母比较 ===")
print("项目\t\t解析解\t\t数值解\t\t差异")
print("-" * 50)

analytical = option.greek
numerical = option.greek_differential

for key in ['delta', 'gamma', 'vega', 'theta', 'rho']:
    diff = abs(analytical[key] - numerical[key])
    print(f"{key}\t\t{analytical[key]:.6f}\t{numerical[key]:.6f}\t{diff:.2e}")
```

### 示例4: 隐含波动率计算

```python
from BSM import BSM, IV, newton_iv
import numpy as np

# 创建基准期权
base_option = BSM("C", 100, 100, 0.25, 1, 0.05, 0.05)
true_vol = 0.25
market_price = base_option.price

print(f"真实波动率: {true_vol:.4f}")
print(f"市场价格: {market_price:.4f}")

# 使用不同方法求隐含波动率
iv_bisection = IV(market_price, "C", 100, 100, 1, 0.05, 0.05)
iv_newton = newton_iv(market_price, "C", 100, 100, 1, 0.05, 0.05)

print(f"隐含波动率 (二分法): {iv_bisection:.4f}")
print(f"隐含波动率 (牛顿法): {iv_newton:.4f}")
print(f"二分法误差: {abs(iv_bisection - true_vol):.6f}")
print(f"牛顿法误差: {abs(iv_newton - true_vol):.6f}")

# 不同行权价的隐含波动率微笑
strikes = np.linspace(90, 110, 11)
implied_vols = []

for K in strikes:
    option_price = BSM("C", 100, K, 0.25, 1, 0.05, 0.05).price
    iv = IV(option_price, "C", 100, K, 1, 0.05, 0.05)
    implied_vols.append(iv)

plt.figure(figsize=(10, 6))
plt.plot(strikes, implied_vols, 'bo-', linewidth=2, markersize=6)
plt.axhline(y=true_vol, color='r', linestyle='--', label=f'真实波动率 ({true_vol:.2f})')
plt.xlabel('行权价格')
plt.ylabel('隐含波动率')
plt.title('隐含波动率微笑')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

### 示例5: 不同市场模型比较

```python
import pandas as pd

# 基础参数
base_params = {
    "CP": "C",
    "S": 100,
    "X": 100,
    "sigma": 0.2,
    "T": 1,
    "r": 0.05
}

# 不同市场模型
models = [
    {"name": "无股利模型", "b": 0.05, "desc": "b = r"},
    {"name": "期货期权", "b": 0.00, "desc": "b = 0"},
    {"name": "股利模型", "b": 0.03, "desc": "b = r - q (q=2%)"},
    {"name": "外汇期权", "b": 0.02, "desc": "b = r - rf (rf=3%)"},
]

results = []
for model in models:
    option = BSM(b=model["b"], **base_params)
    results.append({
        "模型": model["name"],
        "描述": model["desc"],
        "期权价格": round(option.price, 4),
        "Delta": round(option.greek["delta"], 4),
        "Gamma": round(option.greek["gamma"], 4),
        "Vega": round(option.greek["vega"], 4),
        "Theta": round(option.greek["theta"], 4),
        "Rho": round(option.greek["rho"], 4)
    })

df = pd.DataFrame(results)
print("=== 不同市场模型比较 ===")
print(df.to_string(index=False))
```

### 示例6: 期权组合分析

```python
# 构建跨式期权组合 (Straddle)
S = 100
K = 100
sigma = 0.2
T = 1
r = 0.05
b = 0.05

call = BSM("C", S, K, sigma, T, r, b)
put = BSM("P", S, K, sigma, T, r, b)

print("=== 跨式期权组合分析 ===")
print(f"看涨期权价格: {call.price:.4f}")
print(f"看跌期权价格: {put.price:.4f}")
print(f"组合总价格: {call.price + put.price:.4f}")

# 组合希腊字母
print(f"\n组合Delta: {call.greek['delta'] + put.greek['delta']:.4f}")
print(f"组合Gamma: {call.greek['gamma'] + put.greek['gamma']:.4f}")
print(f"组合Vega: {call.greek['vega'] + put.greek['vega']:.4f}")
print(f"组合Theta: {call.greek['theta'] + put.greek['theta']:.4f}")

# 绘制组合损益图
spot_range = np.linspace(80, 120, 100)
portfolio_values = []

for spot in spot_range:
    call_value = max(spot - K, 0)
    put_value = max(K - spot, 0)
    portfolio_value = call_value + put_value - (call.price + put.price)
    portfolio_values.append(portfolio_value)

plt.figure(figsize=(10, 6))
plt.plot(spot_range, portfolio_values, 'b-', linewidth=2)
plt.axhline(y=0, color='r', linestyle='--', alpha=0.7)
plt.axvline(x=K, color='g', linestyle='--', alpha=0.7, label=f'行权价 ({K})')
plt.xlabel('标的价格')
plt.ylabel('损益')
plt.title('跨式期权组合损益图')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

## 理论背景

### Black-Scholes-Merton公式

对于欧式看涨期权：

```
C = S * e^((b-r)*T) * N(d1) - X * e^(-r*T) * N(d2)
```

对于欧式看跌期权：

```
P = X * e^(-r*T) * N(-d2) - S * e^((b-r)*T) * N(-d1)
```

其中：

```
d1 = [ln(S/X) + (b + σ²/2)*T] / (σ*√T)
d2 = d1 - σ*√T
```

### 希腊字母公式

- **Delta (Δ)**: 期权价格对标的价格的敏感性
- **Gamma (Γ)**: Delta对标的价格的敏感性
- **Vega (ν)**: 期权价格对波动率的敏感性
- **Theta (Θ)**: 期权价格对时间的敏感性
- **Rho (ρ)**: 期权价格对利率的敏感性

## 注意事项

1. **模型假设**: BSM模型假设波动率恒定、无交易成本、可连续交易等
2. **适用范围**: 主要适用于欧式期权，美式期权需要其他方法
3. **参数有效性**: 确保所有参数为正数，波动率和利率通常小于1
4. **数值稳定性**: 对于极端参数可能出现数值不稳定
5. **隐含波动率**: 当期权深度价外或价内时，可能无法收敛

## 性能说明

- **解析解**: 计算速度快，精度高，推荐日常使用
- **数值解**: 适用于复杂期权或验证解析解
- **隐含波动率**: 牛顿法通常比二分法更快，但需要合理的初始值

## 扩展应用

该BSM实现可以作为基础模块，扩展到：

- 美式期权定价（结合数值方法）
- 奇异期权定价
- 期权组合风险管理
- 波动率交易策略
- 期权做市商系统

---

*最后更新: 20251009年*
