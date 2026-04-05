#Black-Scholes Model
import numpy as np
from scipy.stats import norm

def black_scholes_call(S, K, T, r, sigma):
    """
    S: 目前股價
    K: 履約價
    T: 到期時間 (年)
    r: 無風險利率 (如 0.02 代表 2%)
    sigma: 波動率 (如 0.2 代表 20%)
    """
    # 1. 計算 d1 與 d2
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    # 2. 帶入公式計算 Call 價格
    # norm.cdf 是累積分布函數，即 N(d)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    
    return call_price, d1, d2

# 範例測試：股價100, 履約價105, 半年後到期, 利率2%, 波動率30%
price, d1, d2 = black_scholes_call(100, 105, 0.5, 0.02, 0.3)
print(f"這張 Call 的理論價格為: {price:.2f}")