#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <random>

// 1. 定義 PayOff 基底類別，方便未來擴充 (如數位期權、障礙期權)
class PayOff {
public:
    virtual double operator()(double spot) const = 0;
    virtual ~PayOff() {}
};

class PayOffCall : public PayOff {
private:
    double strike;
public:
    PayOffCall(double strike_) : strike(strike_) {}
    virtual double operator()(double spot) const override {
        return std::max(spot - strike, 0.0);
    }
};

class PayOffPut : public PayOff {
private:
    double strike;
public:
    PayOffPut(double strike_) : strike(strike_) {}
    virtual double operator()(double spot) const override {
        return std::max(strike - spot, 0.0);
    }
};

// 2. 封裝期權物件
class VanillaOption {
public:
    double expiry;
    const PayOff& payoff;
    VanillaOption(double expiry_, const PayOff& payoff_) 
        : expiry(expiry_), payoff(payoff_) {}
};

// 3. 蒙地卡羅定價引擎
double SimpleMonteCarlo(const VanillaOption& option, 
                        double spot, 
                        double vol, 
                        double r, 
                        unsigned long paths) {
    
    double expiry = option.expiry;
    double variance = vol * vol * expiry;
    double root_variance = std::sqrt(variance);
    double drift = (r - 0.5 * vol * vol) * expiry;

    double running_sum = 0;
    
    // 使用 C++11 的隨機數產生器 (效能優於 rand())
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> dist(0.0, 1.0);

    for (unsigned long i = 0; i < paths; i++) {
        double gauss = dist(generator);
        double this_spot = spot * std::exp(drift + root_variance * gauss);
        double this_payoff = option.payoff(this_spot);
        running_sum += this_payoff;
    }

    double mean = running_sum / paths;
    mean *= std::exp(-r * expiry); // 折現回現在價值
    return mean;
}

int main() {
    // 參數設定
    double spot = 100.0;
    double strike = 105.0;
    double r = 0.02;
    double vol = 0.30;
    double expiry = 0.5;
    unsigned long paths = 100000;

    // 建立一個 Put 期權
    PayOffPut payoff_put(strike);
    VanillaOption my_put(expiry, payoff_put);

    // 執行定價
    double price = SimpleMonteCarlo(my_put, spot, vol, r, paths);

    std::cout << "C++ 模擬出的 Put 價格為: " << price << std::endl;

    return 0;
}