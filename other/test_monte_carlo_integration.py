"""
测试Monte Carlo LSM方法整合的一致性
验证AmericanOption类中的monte_carlo_lsm_price方法与原始Montecarlo-American.py的LS_MC函数输出一致
"""

import numpy as np
import sys
import os

# 添加路径以导入原始文件
sys.path.append(os.path.join(os.path.dirname(__file__), 'other'))

from AmericanOption import AmericanOption

# 导入原始的Monte Carlo函数
def geo_brownian(S, T, r, sigma, steps, paths):
    """
    生成几何布朗运动路径 (原始函数)
    """
    dt = T / steps
    S_path = np.zeros((steps + 1, paths))
    S_path[0] = S
    
    for step in range(1, steps + 1):
        rn = np.random.standard_normal(paths)
        S_path[step] = S_path[step-1] * np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rn)
    
    return S_path

def LS_MC(CP, S, X, T, r, sigma, steps, paths, random_seed=None):
    """
    使用LSM方法计算美式期权价格 (原始函数)
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # 第一步：生成几何布朗运动的价格路径
    S_path = geo_brownian(S, T, r, sigma, steps, paths)
    dt = T / steps
    cash_flow = np.zeros_like(S_path)  # 现金流矩阵
    df = np.exp(-r * dt)  # 折现因子
    
    # 第二步：计算每个时间节点的期权价值
    if CP == 'C':
        cash_flow[-1] = np.maximum(S_path[-1] - X, 0)  # 最后一期的价值
        exercise_value = np.maximum(S_path - X, 0)
    elif CP == 'P':
        cash_flow[-1] = np.maximum(X - S_path[-1], 0)  # 最后一期的价值
        exercise_value = np.maximum(X - S_path, 0)
    else:
        raise ValueError('CP must be C or P')
    
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


def test_monte_carlo_consistency():
    """测试Monte Carlo LSM方法的一致性"""
    print("=== 测试Monte Carlo LSM方法一致性 ===\n")
    
    # 测试参数
    test_cases = [
        {
            "name": "看跌期权 - 标准参数",
            "params": {
                "option_type": "P",
                "spot_price": 100,
                "strike_price": 99,
                "volatility": 0.2,
                "time_to_expiry": 1,
                "risk_free_rate": 0.03,
                "cost_of_carry": 0.03  # 注意：原始代码使用r作为cost_of_carry
            },
            "mc_params": {
                "steps": 1000,
                "paths": 50000,
                "random_seed": 42
            }
        },
        {
            "name": "看涨期权 - 标准参数",
            "params": {
                "option_type": "C",
                "spot_price": 100,
                "strike_price": 105,
                "volatility": 0.25,
                "time_to_expiry": 0.5,
                "risk_free_rate": 0.05,
                "cost_of_carry": 0.05
            },
            "mc_params": {
                "steps": 500,
                "paths": 30000,
                "random_seed": 123
            }
        },
        {
            "name": "看跌期权 - 深度实值",
            "params": {
                "option_type": "P",
                "spot_price": 80,
                "strike_price": 100,
                "volatility": 0.3,
                "time_to_expiry": 0.25,
                "risk_free_rate": 0.02,
                "cost_of_carry": 0.02
            },
            "mc_params": {
                "steps": 200,
                "paths": 20000,
                "random_seed": 456
            }
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"测试案例 {i}: {test_case['name']}")
        print("-" * 50)
        
        # 创建AmericanOption实例
        option = AmericanOption(**test_case['params'])
        
        # 使用新的整合方法计算价格
        new_price = option.monte_carlo_lsm_price(**test_case['mc_params'])
        
        # 使用原始函数计算价格
        original_price = LS_MC(
            CP=test_case['params']['option_type'],
            S=test_case['params']['spot_price'],
            X=test_case['params']['strike_price'],
            T=test_case['params']['time_to_expiry'],
            r=test_case['params']['risk_free_rate'],  # 原始函数使用r
            sigma=test_case['params']['volatility'],
            **test_case['mc_params']
        )
        
        # 计算差异
        diff = abs(new_price - original_price)
        relative_diff = diff / original_price if original_price != 0 else float('inf')
        
        print(f"新方法价格: {new_price:.6f}")
        print(f"原始方法价格: {original_price:.6f}")
        print(f"绝对差异: {diff:.8f}")
        print(f"相对差异: {relative_diff:.6%}")
        
        # 判断是否通过测试（允许很小的数值误差）
        tolerance = 1e-10  # 非常小的容差，因为使用相同随机种子应该完全一致
        passed = diff < tolerance
        
        if passed:
            print("✅ 测试通过")
        else:
            print("❌ 测试失败")
            all_passed = False
        
        print()
    
    return all_passed


def test_different_parameters():
    """测试不同参数下的一致性"""
    print("=== 测试不同参数下的一致性 ===\n")
    
    # 基础参数
    base_params = {
        "option_type": "P",
        "spot_price": 100,
        "strike_price": 100,
        "volatility": 0.2,
        "time_to_expiry": 1,
        "risk_free_rate": 0.05,
        "cost_of_carry": 0.05
    }
    
    # 测试不同的Monte Carlo参数
    mc_test_cases = [
        {"steps": 100, "paths": 10000, "random_seed": 42},
        {"steps": 500, "paths": 20000, "random_seed": 42},
        {"steps": 1000, "paths": 50000, "random_seed": 42},
    ]
    
    print("测试不同Monte Carlo参数:")
    print("-" * 40)
    
    option = AmericanOption(**base_params)
    
    for i, mc_params in enumerate(mc_test_cases, 1):
        # 新方法
        new_price = option.monte_carlo_lsm_price(**mc_params)
        
        # 原始方法
        original_price = LS_MC(
            CP=base_params['option_type'],
            S=base_params['spot_price'],
            X=base_params['strike_price'],
            T=base_params['time_to_expiry'],
            r=base_params['risk_free_rate'],
            sigma=base_params['volatility'],
            **mc_params
        )
        
        diff = abs(new_price - original_price)
        print(f"参数组 {i} (steps={mc_params['steps']}, paths={mc_params['paths']}):")
        print(f"  新方法: {new_price:.6f}, 原始方法: {original_price:.6f}, 差异: {diff:.8f}")
    
    print()


def test_edge_cases():
    """测试边界情况"""
    print("=== 测试边界情况 ===\n")
    
    edge_cases = [
        {
            "name": "极短期权",
            "params": {
                "option_type": "P",
                "spot_price": 100,
                "strike_price": 100,
                "volatility": 0.2,
                "time_to_expiry": 0.01,  # 很短的到期时间
                "risk_free_rate": 0.05,
                "cost_of_carry": 0.05
            }
        },
        {
            "name": "高波动率",
            "params": {
                "option_type": "C",
                "spot_price": 100,
                "strike_price": 100,
                "volatility": 0.8,  # 高波动率
                "time_to_expiry": 1,
                "risk_free_rate": 0.05,
                "cost_of_carry": 0.05
            }
        }
    ]
    
    mc_params = {"steps": 500, "paths": 20000, "random_seed": 42}
    
    for case in edge_cases:
        print(f"边界情况: {case['name']}")
        print("-" * 30)
        
        option = AmericanOption(**case['params'])
        
        try:
            new_price = option.monte_carlo_lsm_price(**mc_params)
            original_price = LS_MC(
                CP=case['params']['option_type'],
                S=case['params']['spot_price'],
                X=case['params']['strike_price'],
                T=case['params']['time_to_expiry'],
                r=case['params']['risk_free_rate'],
                sigma=case['params']['volatility'],
                **mc_params
            )
            
            diff = abs(new_price - original_price)
            print(f"新方法: {new_price:.6f}")
            print(f"原始方法: {original_price:.6f}")
            print(f"差异: {diff:.8f}")
            print("✅ 计算成功")
            
        except Exception as e:
            print(f"❌ 计算失败: {e}")
        
        print()


if __name__ == "__main__":
    print("Monte Carlo LSM方法整合一致性测试")
    print("=" * 60)
    print()
    
    # 运行主要一致性测试
    consistency_passed = test_monte_carlo_consistency()
    
    # 运行参数测试
    test_different_parameters()
    
    # 运行边界情况测试
    test_edge_cases()
    
    # 总结
    print("=" * 60)
    if consistency_passed:
        print("🎉 所有主要一致性测试通过！Monte Carlo LSM方法整合成功。")
    else:
        print("⚠️  部分测试未通过，需要检查实现。")
    
    print("\n测试完成。")