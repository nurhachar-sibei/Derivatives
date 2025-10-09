# 美式期权定价类 (AmericanOption) 使用说明

## 概述

`AmericanOption` 类是一个整合了 BAW (Barone-Adesi-Whaley) 近似解法和二叉树方法的美式期权定价工具。该类提供了统一的接口来计算美式期权价格。

## 功能特性

- **BAW近似解法**: 快速计算美式期权的近似价格
- **二叉树方法**: 精确计算美式期权价格（可调节精度）
- **欧式期权定价**: 支持Black-Scholes-Merton模型
- **方法比较**: 可视化比较不同定价方法的结果
- **完全兼容**: 与原始BAW.py和Binarytree.py代码结果完全一致

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
    option_type="P",        # "C"为看涨期权，"P"为看跌期权
    spot_price=100,         # 标的资产当前价格
    strike_price=99,        # 行权价格
    volatility=0.2,         # 波动率 (20%)
    time_to_expiry=1,       # 到期时间 (1年)
    risk_free_rate=0.03,    # 无风险利率 (3%)
    cost_of_carry=0.03      # 持有成本 (3%)
)

# 计算期权价格
baw_price = option.baw_price()              # BAW方法
binomial_price = option.binomial_tree_price()  # 二叉树方法
european_price = option.bsm_european_price()   # 欧式期权价格

print(f"BAW价格: {baw_price:.4f}")
print(f"二叉树价格: {binomial_price:.4f}")
print(f"欧式期权价格: {european_price:.4f}")
```

### 使用统一接口

```python
# 使用统一的price方法
baw_price = option.price(method="baw")
binomial_price = option.price(method="binomial", steps=1000)

print(f"BAW价格: {baw_price:.4f}")
print(f"二叉树价格: {binomial_price:.4f}")
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

#### 3. 欧式期权定价

```python
bsm_european_price(spot=None, strike=None, vol=None, 
                   time=None, rate=None, carry=None)
```

**参数:** 所有参数可选，如不提供则使用实例属性

**返回:** float - 欧式期权价格

#### 4. 统一定价接口

```python
price(method="baw", **kwargs)
```

**参数:**

- `method` (str): 定价方法
  - `"baw"`: BAW方法
  - `"binomial"`: 二叉树方法
- `**kwargs`: 传递给具体方法的参数

#### 5. 方法比较

```python
compare_methods(time_array=None, plot=True)
```

**参数:**

- `time_array` (array-like): 时间数组，默认为0.01到2年的50个点
- `plot` (bool): 是否绘制比较图

**返回:** dict - 包含各方法价格的字典

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

print("时间\tBAW\t二叉树\t欧式")
for i in range(0, len(results['time']), 5):  # 每5个点打印一次
    t = results['time'][i]
    baw = results['baw'][i]
    binomial = results['binomial'][i]
    european = results['european'][i]
    print(f"{t:.2f}\t{baw:.4f}\t{binomial:.4f}\t{european:.4f}")
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

### 示例4: 批量计算

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
    european_price = option.bsm_european_price()
  
    results.append({
        "类型": params["type"],
        "标的价格": params["S"],
        "行权价格": params["K"],
        "BAW价格": baw_price,
        "二叉树价格": binomial_price,
        "欧式价格": european_price,
        "早行权价值": max(baw_price - european_price, 0)
    })

df = pd.DataFrame(results)
print(df.round(4))
```

## 性能说明

### 计算速度比较

- **BAW方法**: 最快，适合实时计算和大批量计算
- **二叉树方法**: 较慢，精度可调，适合精确计算

### 精度说明

- **BAW方法**: 近似解，对于大多数情况精度足够
- **二叉树方法**: 精确解，步数越多精度越高
- **推荐设置**:
  - 快速计算: BAW方法
  - 精确计算: 二叉树方法，steps=1000或更高

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

*最后更新: 20251009年*
