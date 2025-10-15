# 美式期权定价类 (AmericanOption) 使用说明

## 概述

`AmericanOption` 类是一个整合了 BAW (Barone-Adesi-Whaley) 近似解法、二叉树方法和Monte Carlo LSM方法的美式期权定价工具。该类提供了统一的接口来计算美式期权价格。

## 功能特性

- **BAW近似解法**: 快速计算美式期权的近似价格
- **二叉树方法**: 精确计算美式期权价格（可调节精度）
- **Monte Carlo LSM方法**: 基于最小二乘蒙特卡洛的美式期权定价
- **欧式期权定价**: 支持Black-Scholes-Merton模型
- **方法比较**: 可视化比较不同定价方法的结果
- **完全兼容**: 与原始BAW.py和Binarytree.py代码结果完全一致
- **随机种子控制**: Monte Carlo方法支持结果重现

## 安装依赖

```bash
pip install numpy scipy matplotlib numba
```

## 快速开始

### 基本用法

```python
from AmericanOption import AmericanOption

# 创建美式看跌期权
option = AmericanOption(
    option_type="P",           # "C" 为看涨期权，"P" 为看跌期权
    spot_price=100,            # 标的资产当前价格
    strike_price=99,           # 行权价格
    volatility=0.2,            # 波动率
    time_to_expiry=1,          # 到期时间（年）
    risk_free_rate=0.03,       # 无风险利率
    cost_of_carry=0            # 持有成本
)

# 使用不同方法计算期权价格
baw_price = option.baw_price()                    # BAW方法
binomial_price = option.binomial_tree_price()    # 二叉树方法
monte_carlo_price = option.monte_carlo_lsm_price() # Monte Carlo LSM方法
european_price = option.bsm_european_price()     # 欧式期权价格

print(f"BAW价格: {baw_price:.4f}")
print(f"二叉树价格: {binomial_price:.4f}")
print(f"Monte Carlo价格: {monte_carlo_price:.4f}")
print(f"欧式期权价格: {european_price:.4f}")
```

### 使用统一接口

```python
# 使用统一的price()方法
baw_price = option.price(method="baw")
binomial_price = option.price(method="binomial", steps=2000)
monte_carlo_price = option.price(method="monte_carlo", steps=1000, paths=50000, random_seed=42)

print(f"BAW: {baw_price:.4f}")
print(f"二叉树: {binomial_price:.4f}")
print(f"Monte Carlo: {monte_carlo_price:.4f}")
```

## 详细API文档

### 类初始化

```python
AmericanOption(option_type, spot_price, strike_price, volatility, 
               time_to_expiry, risk_free_rate, cost_of_carry)
```

**参数说明:**

- `option_type` (str): 期权类型

  - `"C"`: 看涨期权 (Call Option)
  - `"P"`: 看跌期权 (Put Option)
- `spot_price` (float): 标的资产当前价格
- `strike_price` (float): 行权价格
- `volatility` (float): 波动率（年化）
- `time_to_expiry` (float): 到期时间（年）
- `risk_free_rate` (float): 无风险利率（年化）
- `cost_of_carry` (float): 持有成本（年化）

  - 当 `b = r` 时：标准无股利模型
  - 当 `b = 0` 时：Black76模型
  - 当 `b = r - q` 时：支付股利模型（q为股利率）
  - 当 `b = r - rf` 时：外汇期权模型（rf为外币利率）

### 主要方法

#### 1. BAW近似解法

```python
baw_price(optimization_method="newton")
```

**参数:**

- `optimization_method` (str): 优化方法
  - `"newton"`: 牛顿迭代法（默认，更快）
  - 其他值: 使用scipy优化方法

**返回:** float - 美式期权价格

#### 2. 二叉树方法

```python
binomial_tree_price(steps=1000)
```

**参数:**

- `steps` (int): 二叉树步数，默认1000（步数越多精度越高，但计算时间越长）

**返回:** float - 美式期权价格

#### 3. Monte Carlo LSM方法

使用最小二乘蒙特卡洛方法计算美式期权价格。

```python
monte_carlo_price = option.monte_carlo_lsm_price(
    steps=1000,        # 时间步数，默认1000
    paths=50000,       # 模拟路径数，默认50000
    random_seed=42     # 随机种子，用于结果重现
)
```

**参数说明：**
- `steps`: 时间步数，影响计算精度
- `paths`: 模拟路径数，影响收敛性和计算时间
- `random_seed`: 随机种子，设置后可重现相同结果

**特点：**
- 基于Longstaff-Schwartz算法
- 适用于复杂的美式期权定价
- 支持路径依赖期权
- 计算时间较长但精度高

#### 4. 欧式期权定价

计算对应的欧式期权价格，用于比较美式期权的提前行权价值。

```python
european_price = option.bsm_european_price(
    spot=None,      # 标的价格，默认使用实例属性
    strike=None,    # 行权价格，默认使用实例属性
    vol=None,       # 波动率，默认使用实例属性
    time=None,      # 到期时间，默认使用实例属性
    rate=None,      # 无风险利率，默认使用实例属性
    carry=None      # 持有成本，默认使用实例属性
)
```

**返回:** float - 欧式期权价格

#### 5. 统一定价接口

提供统一的接口来调用不同的定价方法。

```python
price = option.price(
    method="baw",     # 定价方法: "baw", "binomial", "monte_carlo"
    **kwargs          # 传递给具体方法的参数
)
```

**支持的方法：**
- `"baw"`: BAW近似解法
- `"binomial"`: 二叉树方法
- `"monte_carlo"`, `"montecarlo"`, `"lsm"`: Monte Carlo LSM方法

#### 6. 方法比较

比较不同定价方法在不同到期时间下的结果。

```python
results = option.compare_methods(
    time_array=None,    # 时间数组，默认0.01到2年的50个点
    plot=True,          # 是否绘制比较图
    mc_params=None      # Monte Carlo方法参数
)
```

**参数说明：**
- `time_array`: 用于比较的时间点数组
- `plot`: 是否显示可视化图表
- `mc_params`: Monte Carlo方法的参数字典，包含steps, paths, random_seed

**返回:** dict - 包含各方法价格的字典，键为'time', 'baw', 'binomial', 'monte_carlo', 'european'

## 使用示例

### 示例1: 基本计算

```python
from AmericanOption import AmericanOption

# 创建美式看涨期权
call_option = AmericanOption(
    option_type="C",
    spot_price=100,
    strike_price=105,
    volatility=0.25,
    time_to_expiry=0.5,
    risk_free_rate=0.05,
    cost_of_carry=0.05
)

print(call_option)  # 打印期权信息
print(f"期权价格: {call_option.price():.4f}")
```

### 示例2: 比较不同方法

```python
# 创建美式看跌期权
put_option = AmericanOption("P", 95, 100, 0.3, 1, 0.04, 0.02)

# 比较不同方法的结果
results = put_option.compare_methods()

print("时间\tBAW\t二叉树\tMonte Carlo\t欧式")
for i in range(0, len(results['time']), 5):  # 每5个点打印一次
    t = results['time'][i]
    baw = results['baw'][i]
    binomial = results['binomial'][i]
    monte_carlo = results['monte_carlo'][i]
    european = results['european'][i]
    print(f"{t:.2f}\t{baw:.4f}\t{binomial:.4f}\t{monte_carlo:.4f}\t{european:.4f}")
```

### 示例3: 参数敏感性分析

```python
import numpy as np
import matplotlib.pyplot as plt

# 基础参数
base_option = AmericanOption("P", 100, 100, 0.2, 1, 0.05, 0.05)

# 分析波动率敏感性
volatilities = np.linspace(0.1, 0.5, 20)
prices = []

for vol in volatilities:
    base_option.volatility = vol
    prices.append(base_option.price())

plt.figure(figsize=(10, 6))
plt.plot(volatilities, prices, 'b-', linewidth=2)
plt.xlabel('波动率')
plt.ylabel('期权价格')
plt.title('美式期权价格对波动率的敏感性')
plt.grid(True, alpha=0.3)
plt.show()
```

### 示例4: Monte Carlo方法示例

```python
# Monte Carlo LSM方法的详细使用
option = AmericanOption(
    option_type="P",
    spot_price=100,
    strike_price=105,
    volatility=0.25,
    time_to_expiry=0.5,
    risk_free_rate=0.05,
    cost_of_carry=0.05
)

# 不同精度的Monte Carlo计算
quick_price = option.monte_carlo_lsm_price(steps=200, paths=10000, random_seed=42)
standard_price = option.monte_carlo_lsm_price(steps=1000, paths=50000, random_seed=42)
high_precision_price = option.monte_carlo_lsm_price(steps=2000, paths=100000, random_seed=42)

print(f"快速计算: {quick_price:.4f}")
print(f"标准计算: {standard_price:.4f}")
print(f"高精度计算: {high_precision_price:.4f}")

# 与其他方法比较
baw_price = option.baw_price()
binomial_price = option.binomial_tree_price(steps=1000)

print(f"\n方法比较:")
print(f"BAW方法: {baw_price:.4f}")
print(f"二叉树方法: {binomial_price:.4f}")
print(f"Monte Carlo方法: {standard_price:.4f}")
```

### 示例5: 批量计算

```python
import pandas as pd

# 批量计算不同参数组合的期权价格
parameters = [
    {"type": "C", "S": 100, "K": 95, "vol": 0.2, "T": 1, "r": 0.05, "b": 0.05},
    {"type": "C", "S": 100, "K": 100, "vol": 0.2, "T": 1, "r": 0.05, "b": 0.05},
    {"type": "C", "S": 100, "K": 105, "vol": 0.2, "T": 1, "r": 0.05, "b": 0.05},
    {"type": "P", "S": 100, "K": 95, "vol": 0.2, "T": 1, "r": 0.05, "b": 0.05},
    {"type": "P", "S": 100, "K": 100, "vol": 0.2, "T": 1, "r": 0.05, "b": 0.05},
    {"type": "P", "S": 100, "K": 105, "vol": 0.2, "T": 1, "r": 0.05, "b": 0.05},
]

results = []
for params in parameters:
    option = AmericanOption(
        params["type"], params["S"], params["K"], 
        params["vol"], params["T"], params["r"], params["b"]
    )
  
    baw_price = option.baw_price()
    binomial_price = option.binomial_tree_price()
    monte_carlo_price = option.monte_carlo_lsm_price()
    european_price = option.bsm_european_price()
  
    results.append({
        "类型": params["type"],
        "标的价格": params["S"],
        "行权价格": params["K"],
        "BAW价格": baw_price,
        "二叉树价格": binomial_price,
        "Monte Carlo价格": monte_carlo_price,
        "欧式价格": european_price,
        "早行权价值": max(baw_price - european_price, 0)
    })

df = pd.DataFrame(results)
print(df.round(4))
```

### 示例6: 性能比较

```python
import time

# 创建期权
option = AmericanOption("P", 100, 100, 0.2, 1, 0.05, 0.05)

# 测试各方法的计算时间
methods = [
    ("BAW", lambda: option.baw_price()),
    ("二叉树", lambda: option.binomial_tree_price(steps=1000)),
    ("Monte Carlo", lambda: option.monte_carlo_lsm_price(steps=1000, paths=50000, random_seed=42))
]

print("性能比较:")
print("-" * 40)
for name, method in methods:
    start_time = time.time()
    price = method()
    end_time = time.time()
    
    print(f"{name:12s}: {price:.4f} (耗时: {end_time - start_time:.3f}秒)")
```

### 计算速度比较

- **BAW方法**: 最快，适合实时计算和大批量计算
- **二叉树方法**: 较慢，精度可调，适合精确计算
- **Monte Carlo LSM方法**: 最慢，但适用于复杂期权和路径依赖期权

### 精度说明

- **BAW方法**: 近似解，对于大多数情况精度足够
- **二叉树方法**: 精确解，步数越多精度越高
- **Monte Carlo LSM方法**: 统计解，路径数越多精度越高
- **推荐设置**:
  - 快速计算: BAW方法
  - 精确计算: 二叉树方法，steps=1000或更高
  - 复杂期权: Monte Carlo LSM方法，paths=50000或更高

## 注意事项

1. **参数有效性**: 确保所有参数为正数，波动率和利率通常小于1
2. **时间单位**: 所有时间相关参数均为年化值
3. **早行权条件**: 当 `cost_of_carry > risk_free_rate` 时，美式期权价值等于欧式期权价值
4. **数值稳定性**: 对于极端参数（如极短到期时间），可能出现数值不稳定

## 与原始代码的兼容性

该类与原始的 `BAW.py` 和 `Binarytree.py` 完全兼容：

```python
# 原始代码调用方式
from BAW import BAW
from Binarytree import binarytree_am

original_baw = BAW("P", 100, 99, 0.2, 1, 0.03, 0.03)
original_binomial = binarytree_am("P", 1000, 100, 1, 0.2, 99, 0.03, 0.03)

# 新类调用方式
option = AmericanOption("P", 100, 99, 0.2, 1, 0.03, 0.03)
new_baw = option.baw_price()
new_binomial = option.binomial_tree_price(1000)

# 结果完全一致
assert abs(original_baw - new_baw) < 1e-10
assert abs(original_binomial - new_binomial) < 1e-10
```

## 测试

运行测试以验证功能正确性：

```bash
python test_american_option.py
```

测试包括：

- BAW方法一致性测试
- 二叉树方法一致性测试
- 欧式期权价格测试
- 统一接口测试
- 与原始代码的完整比较测试

---

*最后更新: 2025年1月9日*