"""
测试美式期权类的功能
验证与原始代码的一致性
"""

import numpy as np
from AmericanOption import AmericanOption
from BAW import BAW, BSM
from Binarytree import binarytree_am


def test_baw_consistency():
    """测试BAW方法的一致性"""
    print("=== 测试BAW方法一致性 ===")
    
    # 测试参数
    test_cases = [
        {"CP": "P", "S": 100, "X": 99, "sigma": 0.2, "T": 1, "r": 0.03, "b": 0.03},
        {"CP": "C", "S": 100, "X": 105, "sigma": 0.25, "T": 0.5, "r": 0.05, "b": 0.05},
        {"CP": "P", "S": 95, "X": 100, "sigma": 0.3, "T": 2, "r": 0.04, "b": 0.02},
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\n测试案例 {i+1}:")
        print(f"参数: {params}")
        
        # 原始BAW函数
        original_price = BAW(
            CP=params["CP"], 
            S=params["S"], 
            X=params["X"], 
            sigma=params["sigma"], 
            T=params["T"], 
            r=params["r"], 
            b=params["b"]
        )
        
        # 新的AmericanOption类
        option = AmericanOption(
            option_type=params["CP"],
            spot_price=params["S"],
            strike_price=params["X"],
            volatility=params["sigma"],
            time_to_expiry=params["T"],
            risk_free_rate=params["r"],
            cost_of_carry=params["b"]
        )
        new_price = option.baw_price()
        
        print(f"原始BAW价格: {original_price:.6f}")
        print(f"新类BAW价格: {new_price:.6f}")
        print(f"差异: {abs(original_price - new_price):.8f}")
        
        # 检查差异是否在可接受范围内
        assert abs(original_price - new_price) < 1e-6, f"BAW价格差异过大: {abs(original_price - new_price)}"
    
    print("\n✓ BAW方法一致性测试通过")


def test_binomial_consistency():
    """测试二叉树方法的一致性"""
    print("\n=== 测试二叉树方法一致性 ===")
    
    # 测试参数
    test_cases = [
        {"CP": "P", "S0": 100, "K": 99, "sigma": 0.2, "T": 1, "r": 0.03, "b": 0.03, "m": 1000},
        {"CP": "C", "S0": 100, "K": 105, "sigma": 0.25, "T": 0.5, "r": 0.05, "b": 0.05, "m": 500},
        {"CP": "P", "S0": 95, "K": 100, "sigma": 0.3, "T": 2, "r": 0.04, "b": 0.02, "m": 800},
    ]
    
    for i, params in enumerate(test_cases):
        print(f"\n测试案例 {i+1}:")
        print(f"参数: {params}")
        
        # 原始二叉树函数
        original_price = binarytree_am(
            CP=params["CP"],
            m=params["m"],
            S0=params["S0"],
            T=params["T"],
            sigma=params["sigma"],
            K=params["K"],
            r=params["r"],
            b=params["b"]
        )
        
        # 新的AmericanOption类
        option = AmericanOption(
            option_type=params["CP"],
            spot_price=params["S0"],
            strike_price=params["K"],
            volatility=params["sigma"],
            time_to_expiry=params["T"],
            risk_free_rate=params["r"],
            cost_of_carry=params["b"]
        )
        new_price = option.binomial_tree_price(steps=params["m"])
        
        print(f"原始二叉树价格: {original_price:.6f}")
        print(f"新类二叉树价格: {new_price:.6f}")
        print(f"差异: {abs(original_price - new_price):.8f}")
        
        # 检查差异是否在可接受范围内
        assert abs(original_price - new_price) < 1e-6, f"二叉树价格差异过大: {abs(original_price - new_price)}"
    
    print("\n✓ 二叉树方法一致性测试通过")


def test_european_option():
    """测试欧式期权价格计算"""
    print("\n=== 测试欧式期权价格计算 ===")
    
    # 测试参数
    params = {"CP": "C", "S": 100, "X": 100, "sigma": 0.2, "T": 1, "r": 0.05, "b": 0.05}
    
    # 原始BSM函数
    original_price = BSM(
        CP=params["CP"],
        S=params["S"],
        X=params["X"],
        sigma=params["sigma"],
        T=params["T"],
        r=params["r"],
        b=params["b"]
    )
    
    # 新的AmericanOption类
    option = AmericanOption(
        option_type=params["CP"],
        spot_price=params["S"],
        strike_price=params["X"],
        volatility=params["sigma"],
        time_to_expiry=params["T"],
        risk_free_rate=params["r"],
        cost_of_carry=params["b"]
    )
    new_price = option.bsm_european_price()
    
    print(f"原始BSM价格: {original_price:.6f}")
    print(f"新类BSM价格: {new_price:.6f}")
    print(f"差异: {abs(original_price - new_price):.8f}")
    
    assert abs(original_price - new_price) < 1e-10, f"BSM价格差异过大: {abs(original_price - new_price)}"
    print("✓ 欧式期权价格计算测试通过")


def test_price_interface():
    """测试统一价格接口"""
    print("\n=== 测试统一价格接口 ===")
    
    option = AmericanOption(
        option_type="P",
        spot_price=100,
        strike_price=99,
        volatility=0.2,
        time_to_expiry=1,
        risk_free_rate=0.03,
        cost_of_carry=0.03
    )
    
    # 测试不同方法
    baw_price1 = option.price(method="baw")
    baw_price2 = option.baw_price()
    
    binomial_price1 = option.price(method="binomial", steps=1000)
    binomial_price2 = option.binomial_tree_price(steps=1000)
    
    print(f"BAW价格 (接口): {baw_price1:.6f}")
    print(f"BAW价格 (直接): {baw_price2:.6f}")
    print(f"二叉树价格 (接口): {binomial_price1:.6f}")
    print(f"二叉树价格 (直接): {binomial_price2:.6f}")
    
    assert abs(baw_price1 - baw_price2) < 1e-10, "BAW接口不一致"
    assert abs(binomial_price1 - binomial_price2) < 1e-10, "二叉树接口不一致"
    
    print("✓ 统一价格接口测试通过")


def test_comparison_with_original():
    """与原始代码的完整比较测试"""
    print("\n=== 与原始代码的完整比较测试 ===")
    
    # 复现原始代码中的比较
    T_array = np.linspace(0.01, 2, 10)  # 使用较少的点进行快速测试
    
    print("时间\t原BAW\t新BAW\t原二叉树\t新二叉树\tBAW差异\t二叉树差异")
    print("-" * 80)
    
    max_baw_diff = 0
    max_binomial_diff = 0
    
    for T in T_array:
        # 原始方法
        original_baw = BAW(CP="P", S=100, X=99, sigma=0.2, T=T, r=0.03, b=0.03)
        original_binomial = binarytree_am(CP="P", m=500, S0=100, T=T, sigma=0.2, K=99, r=0.03, b=0.03)
        
        # 新类方法
        option = AmericanOption("P", 100, 99, 0.2, T, 0.03, 0.03)
        new_baw = option.baw_price()
        new_binomial = option.binomial_tree_price(steps=500)
        
        baw_diff = abs(original_baw - new_baw)
        binomial_diff = abs(original_binomial - new_binomial)
        
        max_baw_diff = max(max_baw_diff, baw_diff)
        max_binomial_diff = max(max_binomial_diff, binomial_diff)
        
        print(f"{T:.2f}\t{original_baw:.4f}\t{new_baw:.4f}\t{original_binomial:.4f}\t{new_binomial:.4f}\t{baw_diff:.2e}\t{binomial_diff:.2e}")
    
    print(f"\n最大BAW差异: {max_baw_diff:.2e}")
    print(f"最大二叉树差异: {max_binomial_diff:.2e}")
    
    assert max_baw_diff < 1e-6, f"BAW最大差异过大: {max_baw_diff}"
    assert max_binomial_diff < 1e-6, f"二叉树最大差异过大: {max_binomial_diff}"
    
    print("✓ 完整比较测试通过")


def run_all_tests():
    """运行所有测试"""
    print("开始运行美式期权类测试...")
    
    try:
        test_european_option()
        test_baw_consistency()
        test_binomial_consistency()
        test_price_interface()
        test_comparison_with_original()
        
        print("\n" + "="*50)
        print("🎉 所有测试通过！新的AmericanOption类功能正确！")
        print("="*50)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()