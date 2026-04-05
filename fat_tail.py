import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, t

# 參數設定
S0 = 100       # 初始股價
mu = 0.05      # 年化預期報酬
sigma = 0.2    # 年化波動率
T = 1.0        # 1 年
steps = 252
dt = T / steps
n_paths = 5000 # 模擬 5000 條路徑看分佈

# 1. 常態分佈路徑 (Standard GBM)
np.random.seed(42)
z_norm = np.random.normal(0, 1, (steps, n_paths))
paths_norm = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_norm, axis=0))

# 2. 肥尾分佈路徑 (使用 Student's t-distribution)
# df (自由度) 越小，尾巴越肥。df=3 是常見的金融模擬設定
df = 3 
# 注意：t 分佈需要縮放，使其變異數與常態分佈一致，以便公平比較
z_t = np.random.standard_t(df, (steps, n_paths)) * np.sqrt((df-2)/df)
paths_t = S0 * np.exp(np.cumsum((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z_t, axis=0))

# 3. 視覺化對比
plt.figure(figsize=(12, 6))

# 畫出最後一天的股價分佈直方圖
plt.hist(paths_norm[-1], bins=100, alpha=0.5, label='Normal (BSM)', density=True, color='blue')
plt.hist(paths_t[-1], bins=100, alpha=0.5, label='Fat-Tail (Student-t)', density=True, color='red')

plt.title(f"Price Distribution at T=1: Normal vs Fat-Tail (df={df})")
plt.xlabel("Stock Price")
plt.ylabel("Probability Density")
plt.legend()
plt.show()

# 4. 計算極端事件機率 (例如股價跌破 50 元)
crash_norm = np.mean(paths_norm[-1] < 50)
crash_t = np.mean(paths_t[-1] < 50)

print(f"常態分佈下，跌破 50 元的機率: {crash_norm:.2%}")
print(f"肥尾分佈下，跌破 50 元的機率: {crash_t:.2%}")