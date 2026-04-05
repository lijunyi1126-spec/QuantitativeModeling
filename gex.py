import numpy as np

def calculate_gex(normalized_gamma, open_interest, spot_price):
    # 簡化模型：假設造市商在 Call 端是 Short，在 Put 端是 Short (這取決於市場結構)
    # GEX = Gamma * Open_Interest * (Spot_Price^2) * 0.01
    # 這裡的 0.01 是指股價變動 1% 時，造市商需要調整多少金額的部位
    gex_value = normalized_gamma * open_interest * (spot_price**2) * 0.01
    return gex_value

# 當全市場總和的 GEX 轉負時，代表「反身性崩盤」的風險極高