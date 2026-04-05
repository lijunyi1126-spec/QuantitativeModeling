#Sensitivity Analysis
import numpy as np
import matplotlib.pyplot as plt

# 模擬參數
S0 = 100
n_days = 120
np.random.seed(42)

# 模擬一個震盪後大跌的行情
daily_ret = np.random.normal(0.0001, 0.015, n_days)
daily_ret[80:100] = -0.04 # 第 80 天開始崩盤
price_path = S0 * np.cumprod(1 + daily_ret)

# 設定三種不同的保險方案 (假設權利金由 BSM 概算)
# 方案 A: 履約價 100 (ATM), 權利金 4.0
# 方案 B: 履約價 90 (OTM), 權利金 1.5
# 方案 C: 無保險, 權利金 0

def get_portfolio_value(s_path, k, premium):
    put_payoff = np.maximum(k - s_path, 0)
    return s_path + put_payoff - premium

val_no_ins = price_path
val_atm = get_portfolio_value(price_path, 100, 4.0)
val_otm = get_portfolio_value(price_path, 90, 1.5)

# 繪圖
plt.figure(figsize=(12, 7))
plt.plot(val_no_ins, label='No Insurance', color='gray', linestyle='--')
plt.plot(val_atm, label='ATM Protection (K=100, Premium=4.0)', color='blue', linewidth=2)
plt.plot(val_otm, label='OTM Protection (K=90, Premium=1.5)', color='orange', linewidth=2)

plt.axhline(100-4.0, color='blue', alpha=0.2, linestyle=':')
plt.axhline(90-1.5, color='orange', alpha=0.2, linestyle=':')

plt.title("Comparison of Different Hedging Depths")
plt.ylabel("Asset Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()