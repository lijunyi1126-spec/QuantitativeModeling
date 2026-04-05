import numpy as np
import matplotlib.pyplot as plt

# 1. 參數設定
S0 = 100          # 初始股價
T = 1.0           # 模擬時間（1年）
mu = 0.05         # 年化預期報酬率 (Drift)
sigma = 0.2       # 年化波動率 (Volatility, 20%)
steps = 252       # 總步數（一年約 252 個交易日）
dt = T / steps    # 每一步的時間間隔
n_paths = 10      # 想要模擬的路徑數量

# 2. 建立隨機矩陣
# 產生 (steps, n_paths) 的標準正態分佈隨機數 epsilon
np.random.seed(42) # 固定隨機種子，方便實驗
z = np.random.normal(0, 1, (steps, n_paths))

# 3. 計算股價路徑
# 使用 GBM 公式：S(t+1) = S(t) * exp((mu - 0.5 * sigma^2) * dt + sigma * sqrt(dt) * z)
# 我們先計算每日變動率，再用累積乘積 (cumprod) 算出股價
drift = (mu - 0.5 * sigma**2) * dt
diffusion = sigma * np.sqrt(dt) * z
daily_returns = np.exp(drift + diffusion)

# 將初始價格放第一行，並計算路徑
paths = np.zeros((steps + 1, n_paths))
paths[0] = S0
paths[1:] = S0 * np.cumprod(daily_returns, axis=0)

# 4. 繪圖
plt.figure(figsize=(10, 6))
plt.plot(paths)
plt.axhline(S0, color='red', linestyle='--', alpha=0.5, label='Start Price')
plt.title(f"GBM Monte Carlo Simulation ({n_paths} paths)")
plt.xlabel("Days")
plt.ylabel("Stock Price")
plt.legend(['Paths', 'Initial Price'])
plt.grid(True, alpha=0.3)
plt.show()