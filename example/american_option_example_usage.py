"""
美式期权类使用示例
演示AmericanOption类的各种功能
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from AmericanOption import AmericanOption

def example_1_basic_usage():
    """示例1: 基本使用方法"""
    print("=" * 50)
    print("示例1: 基本使用方法")
    print("=" * 50)
    
    # 创建美式看跌期权
    option = AmericanOption(
        option_type="P",        # 看跌期权
        spot_price=100,         # 标的价格100
        strike_price=99,        # 行权价格99
        volatility=0.2,         # 波动率20%
        time_to_expiry=1,       # 1年到期
        risk_free_rate=0.03,    # 无风险利率3%
        cost_of_carry=0.03      # 持有成本3%
    )
    
    print(option)
    
    # 计算期权价格
    baw_price = option.baw_price()
    binomial_price = option.binomial_tree_price()
    european_price = option.bsm_european_price()
    
    print(f"BAW价格: {baw_price:.4f}")
    print(f"二叉树价格: {binomial_price:.4f}")
    print(f"欧式期权价格: {european_price:.4f}")
    print(f"早行权价值: {baw_price - european_price:.4f}")
    
    # 使用统一接口
    print(f"\n使用统一接口:")
    print(f"BAW价格: {option.price(method='baw'):.4f}")
    print(f"二叉树价格: {option.price(method='binomial', steps=500):.4f}")


def example_2_compare_methods():
    """示例2: 比较不同定价方法"""
    print("\n" + "=" * 50)
    print("示例2: 比较不同定价方法")
    print("=" * 50)
    
    # 创建美式看跌期权
    option = AmericanOption("P", 100, 105, 0.25, 1, 0.05, 0.03)
    
    # 比较不同方法，并绘制图表
    results = option.compare_methods(time_array=np.linspace(0.1, 2, 20), plot=True)
    
    # 打印部分结果
    print("\n时间\tBAW\t二叉树\t欧式\t早行权价值")
    print("-" * 50)
    for i in range(0, len(results['time']), 4):
        t = results['time'][i]
        baw = results['baw'][i]
        binomial = results['binomial'][i]
        european = results['european'][i]
        early_exercise = baw - european
        print(f"{t:.2f}\t{baw:.3f}\t{binomial:.3f}\t{european:.3f}\t{early_exercise:.3f}")


def example_3_sensitivity_analysis():
    """示例3: 参数敏感性分析"""
    print("\n" + "=" * 50)
    print("示例3: 参数敏感性分析")
    print("=" * 50)
    
    # 基础期权
    base_option = AmericanOption("C", 100, 100, 0.2, 1, 0.05, 0.05)
    
    # 1. 波动率敏感性
    volatilities = np.linspace(0.1, 0.5, 20)
    vol_prices = []
    
    for vol in volatilities:
        base_option.volatility = vol
        vol_prices.append(base_option.price())
    
    # 2. 标的价格敏感性
    base_option.volatility = 0.2  # 重置波动率
    spot_prices = np.linspace(80, 120, 20)
    spot_price_values = []
    
    for spot in spot_prices:
        base_option.spot_price = spot
        spot_price_values.append(base_option.price())
    
    # 绘制敏感性分析图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 波动率敏感性
    ax1.plot(volatilities, vol_prices, 'b-', linewidth=2, marker='o')
    ax1.set_xlabel('波动率')
    ax1.set_ylabel('期权价格')
    ax1.set_title('期权价格对波动率的敏感性')
    ax1.grid(True, alpha=0.3)
    
    # 标的价格敏感性
    ax2.plot(spot_prices, spot_price_values, 'r-', linewidth=2, marker='s')
    ax2.set_xlabel('标的价格')
    ax2.set_ylabel('期权价格')
    ax2.set_title('期权价格对标的价格的敏感性')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print("敏感性分析图已生成")


def example_4_batch_calculation():
    """示例4: 批量计算"""
    print("\n" + "=" * 50)
    print("示例4: 批量计算不同期权")
    print("=" * 50)
    
    # 定义不同的期权参数
    parameters = [
        {"type": "C", "S": 100, "K": 95, "vol": 0.2, "T": 0.5, "desc": "浅度价内看涨"},
        {"type": "C", "S": 100, "K": 100, "vol": 0.2, "T": 0.5, "desc": "平价看涨"},
        {"type": "C", "S": 100, "K": 105, "vol": 0.2, "T": 0.5, "desc": "价外看涨"},
        {"type": "P", "S": 100, "K": 95, "vol": 0.2, "T": 0.5, "desc": "价外看跌"},
        {"type": "P", "S": 100, "K": 100, "vol": 0.2, "T": 0.5, "desc": "平价看跌"},
        {"type": "P", "S": 100, "K": 105, "vol": 0.2, "T": 0.5, "desc": "浅度价内看跌"},
    ]
    
    results = []
    for params in parameters:
        option = AmericanOption(
            params["type"], params["S"], params["K"], 
            params["vol"], params["T"], 0.05, 0.03
        )
        
        baw_price = option.baw_price()
        binomial_price = option.binomial_tree_price(steps=500)
        european_price = option.bsm_european_price()
        
        results.append({
            "描述": params["desc"],
            "类型": params["type"],
            "标的价格": params["S"],
            "行权价格": params["K"],
            "BAW价格": round(baw_price, 4),
            "二叉树价格": round(binomial_price, 4),
            "欧式价格": round(european_price, 4),
            "早行权价值": round(max(baw_price - european_price, 0), 4),
            "价格差异": round(abs(baw_price - binomial_price), 6)
        })
    
    # 创建DataFrame并打印
    df = pd.DataFrame(results)
    print(df.to_string(index=False))


def example_5_different_models():
    """示例5: 不同市场模型"""
    print("\n" + "=" * 50)
    print("示例5: 不同市场模型的期权定价")
    print("=" * 50)
    
    base_params = {
        "option_type": "P",
        "spot_price": 100,
        "strike_price": 100,
        "volatility": 0.25,
        "time_to_expiry": 1,
        "risk_free_rate": 0.05
    }
    
    models = [
        {"name": "无股利模型", "b": 0.05, "desc": "b = r"},
        {"name": "Black76模型", "b": 0.00, "desc": "b = 0"},
        {"name": "支付股利模型", "b": 0.03, "desc": "b = r - q (q=2%)"},
        {"name": "外汇期权模型", "b": 0.02, "desc": "b = r - rf (rf=3%)"},
    ]
    
    print("模型\t\t描述\t\t\tBAW价格\t二叉树价格\t欧式价格")
    print("-" * 80)
    
    for model in models:
        option = AmericanOption(cost_of_carry=model["b"], **base_params)
        
        baw_price = option.baw_price()
        binomial_price = option.binomial_tree_price()
        european_price = option.bsm_european_price()
        
        print(f"{model['name']:<12}\t{model['desc']:<20}\t{baw_price:.4f}\t\t{binomial_price:.4f}\t\t{european_price:.4f}")


def example_6_performance_comparison():
    """示例6: 性能比较"""
    print("\n" + "=" * 50)
    print("示例6: 计算性能比较")
    print("=" * 50)
    
    import time
    
    option = AmericanOption("P", 100, 100, 0.2, 1, 0.05, 0.03)
    
    # BAW方法性能测试
    start_time = time.time()
    for _ in range(1000):
        option.baw_price()
    baw_time = time.time() - start_time
    
    # 二叉树方法性能测试（不同步数）
    steps_list = [100, 500, 1000, 2000]
    binomial_times = []
    binomial_prices = []
    
    for steps in steps_list:
        start_time = time.time()
        for _ in range(100):  # 减少次数因为二叉树较慢
            price = option.binomial_tree_price(steps=steps)
        binomial_time = time.time() - start_time
        binomial_times.append(binomial_time)
        binomial_prices.append(price)
    
    print(f"BAW方法 (1000次): {baw_time:.4f}秒")
    print(f"BAW单次平均: {baw_time/1000*1000:.4f}毫秒")
    print()
    
    print("二叉树方法 (100次):")
    for i, steps in enumerate(steps_list):
        avg_time = binomial_times[i] / 100 * 1000  # 转换为毫秒
        print(f"  {steps}步: {binomial_times[i]:.4f}秒, 单次平均: {avg_time:.2f}毫秒, 价格: {binomial_prices[i]:.6f}")


def main():
    """运行所有示例"""
    print("美式期权类 (AmericanOption) 使用示例")
    print("=" * 60)
    
    try:
        example_1_basic_usage()
        example_2_compare_methods()
        example_3_sensitivity_analysis()
        example_4_batch_calculation()
        example_5_different_models()
        example_6_performance_comparison()
        
        print("\n" + "=" * 60)
        print("🎉 所有示例运行完成！")
        print("=" * 60)
        
    except Exception as e:
        print(f"❌ 运行示例时出错: {e}")
        raise


if __name__ == "__main__":
    main()