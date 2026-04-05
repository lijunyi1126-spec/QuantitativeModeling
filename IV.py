#Implied Volatility
import numpy as np
from scipy.stats import norm

# 基礎 BSM Call 價格與 Vega 計算
def bsm_call_and_vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    # Vega 是 BSM 對 sigma 的一階導數
    vega = S * norm.pdf(d1) * np.sqrt(T)
    
    return price, vega

def find_implied_vol(market_price, S, K, T, r, iterations=100, precision=1.0e-5):
    # 初始猜測值 (通常設 0.2 或 0.5)
    sigma = 0.5
    
    for i in range(iterations):
        price, vega = bsm_call_and_vega(S, K, T, r, sigma)
        diff = market_price - price
        
        if abs(diff) < precision:
            return sigma
        
        # 牛頓法更新：sigma = sigma + (價差 / 敏感度)
        sigma = sigma + diff / vega
        
    return sigma

# 測試：假設目前股價 100，履約價 105，半年到期，利率 2%
# 市場上這張期權賣 6.5 元，請問隱含波動率是多少？
iv = find_implied_vol(6.5, 100, 105, 0.5, 0.02)
print(f"市場目前的隱含波動率 (IV) 為: {iv:.2%}")