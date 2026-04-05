import numpy as np
import matplotlib.pyplot as plt

# 1. 模擬參數：設定一個發生黑天鵝大跌的場景
S0 = 100          # 初始股價
K = 95            # Put 的履約價 (想在跌破 5% 時啟動保險)
put_premium = 2.0 # 買這張保險要付的權利金
n_days = 100      # 模擬 100 天

# 模擬一條「先平穩後暴跌」的股價路徑 (黑天鵝)
np.random.seed(10)
returns = np.random.normal(0.0002, 0.01, n_days) # 平時小波動
returns[70:80] = -0.05                           # 第 70 天開始連續暴跌 5%
price_path = S0 * np.cumprod(1 + returns)

# 2. 計算資產價值
# 純持股價值
stock_only = price_path

# 保護性看跌 (Protective Put) 價值
# 總價值 = 股價 + Put價值 - 權利金成本
# Put價值 = max(K - S, 0)
put_payoff = np.maximum(K - price_path, 0)
protected_portfolio = price_path + put_payoff - put_premium

# 3. 繪圖觀察
plt.figure(figsize=(12, 6))
plt.plot(stock_only, label='Stock Only (No Insurance)', color='red', alpha=0.6)
plt.plot(protected_portfolio, label='Stock + Put (Protected)', color='green', linewidth=2)
plt.axhline(K - put_premium, color='black', linestyle='--', label='Floor Level')

plt.title("Protective Put Strategy: Hedging Against Black Swan")
plt.xlabel("Days")
plt.ylabel("Portfolio Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"最終純持股價值: {stock_only[-1]:.2f}")
print(f"最終保護後價值: {protected_portfolio[-1]:.2f}")