import numpy as np
import matplotlib.pyplot as plt

# 共同參數
S0 = 100       # 初始股價
T = 1.0        # 模擬 1 年
dt = 1/252     # 以交易日為單位
steps = int(T/dt)
mu = 0.05      # 預期年化報酬率

# Heston 模型特有參數
v0 = 0.04      # 初始變異數 (相當於波動率 20%)
kappa = 3.0    # 均值回歸速度 (波動率回到平均值的力量)
theta = 0.04   # 長期平均變異數
sigma_v = 0.3  # 波動率的波動 (Vol of Vol)
rho = -0.7     # 股價與波動率的相關性 (負值代表跌時波動大)

# 初始化陣列
prices_h = np.zeros(steps)
vars_h = np.zeros(steps)
prices_h[0], vars_h[0] = S0, v0

# 蒙地卡羅模擬
for t in range(1, steps):
    # 生成相關的隨機數
    z1 = np.random.normal(0, 1)
    z2 = np.random.normal(0, 1)
    zv = rho * z1 + np.sqrt(1 - rho**2) * z2
    
    # 1. 波動率演進 (CIR過程)
    # 加上 max(0, ...) 防止變異數變為負數
    vars_h[t] = vars_h[t-1] + kappa * (theta - vars_h[t-1]) * dt + \
                sigma_v * np.sqrt(max(0, vars_h[t-1]) * dt) * zv
    
    # 2. 股價演進
    prices_h[t] = prices_h[t-1] * np.exp((mu - 0.5 * max(0, vars_h[t-1])) * dt + \
                  np.sqrt(max(0, vars_h[t-1]) * dt) * z1)

# 繪圖
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
ax1.plot(prices_h, label='Heston Price Path', color='blue')
ax1.set_title("Heston Model: Simulated Price Path")
ax1.legend()

ax2.plot(np.sqrt(vars_h), label='Stochastic Volatility', color='orange')
ax2.set_title("Heston Model: Volatility Over Time")
ax2.legend()
plt.show()