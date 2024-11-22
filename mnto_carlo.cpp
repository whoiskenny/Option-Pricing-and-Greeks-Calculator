#include <iostream>
#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

// Constants for Black-scholes Model
const double PI = 3.14159265358979323846;

// Global random number generator
static std::mt19937 generator(std::chrono::high_resolution_clock::now().time_since_epoch().count());

// Function to generate normally distributed random numbers
double generateGaussianNoise(double mean, double stddev) {
    static std::normal_distribution<double> distribution(mean, stddev);
    return distribution(generator);
}

// Cumulative normal distribution
double normalCDF(double x) {
  return 0.5 * std:: erfc(-x / std::sqrt(2.0));
}

//Black-Scholes formula for European options 
double blackScholesPrice(double S, double K, double r, double sigma, double T, bool isCall) { double d1 = (std::log(S / K) + (r * 0.5 * sigma * sigma) * T) / (sigma * std::sqrt(T)); double d2 = d1 - sigma * std::sqrt(T); if (isCall) {
    return S * normalCDF(d1) - K * std::exp(-r * T) * normalCDF(d2);
  } else {
    return K * std::exp(-r * T) * normalCDF(-d2) - S * normalCDF(-d1);
  }
}

// Function to calculate the payoff of a European call option
double callOptionPayoff(double S, double K) {
    return std::max(S - K, 0.0);
}

// Function to calculate the payoff of a European put option
double putOptionPayoff(double S, double K) {
    return std::max(K - S, 0.0);
}

// Monte Carlo Simulation for European option pricing
double monteCarloOptionPricing(double S0, double K, double r, double sigma, double T, int numSimulations, bool isCallOption) {
    double payoffSum = 0.0;
    for (int i = 0; i < numSimulations; ++i) {
        // Generate a random price path
        double ST = S0 * std::exp((r - 0.5 * sigma * sigma) * T + sigma * std::sqrt(T) * generateGaussianNoise(0.0, 1.0));
        // Calculate the payoff for this path
        double payoff = isCallOption ? callOptionPayoff(ST, K) : putOptionPayoff(ST, K);
        // Accumulate the payoff
        payoffSum += payoff;
    }
    // Calculate the average payoff and discount it to present value
    double averagePayoff = payoffSum / static_cast<double>(numSimulations);
    return std::exp(-r * T) * averagePayoff;
}

struct Greeks {
  double delta;
  double gamma; 
  double theta;
  double vega;
  double rho;
};

// Calculate Greeks using finite difference method
// Calculate Greeks using finite difference method
Greeks calculateGreeks(double S, double K, double r, double sigma, double T, bool isCall) {
    const double h = 0.01; // Small change for finite difference
    
    double price = blackScholesPrice(S, K, r, sigma, T, isCall);
    double priceUp = blackScholesPrice(S + h, K, r, sigma, T, isCall);
    double priceDown = blackScholesPrice(S - h, K, r, sigma, T, isCall);
    double priceT = blackScholesPrice(S, K, r, sigma, T - h / 365, isCall);
    double priceSigma = blackScholesPrice(S, K, r, sigma + h, T, isCall);
    double priceR = blackScholesPrice(S, K, r + h, sigma, T, isCall);
    
    Greeks greeks;
    greeks.delta = (priceUp - priceDown) / (2 * h);
    greeks.gamma = (priceUp - 2 * price + priceDown) / (h * h);
    greeks.theta = -(priceT - price) / (h / 365);
    greeks.vega = (priceSigma - price) / h;
    greeks.rho = (priceR - price) / h;
    
    return greeks;
}

// Longstaff-Schwartz algorithm for American option pricing
double americanOptionPricing(double S0, double K, double r, double sigma, double T, int numSimulations, int numSteps, bool isCall) {
    std::vector<double> S(numSimulations);
    std::vector<double> payoff(numSimulations);
    double dt = T / numSteps;
    
    // Generate price paths
    for (int i = 0; i < numSimulations; ++i) {
        S[i] = S0;
        for (int j = 0; j < numSteps; ++j) {
            S[i] *= std::exp((r - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * generateGaussianNoise(0.0, 1.0));
        }
        payoff[i] = isCall ? callOptionPayoff(S[i], K) : putOptionPayoff(S[i], K);
    }
    
    // Backward induction
    for (int j = numSteps - 1; j > 0; --j) {
        std::vector<double> X;
        std::vector<double> Y;
        
        for (int i = 0; i < numSimulations; ++i) {
            double currentPayoff = isCall ? callOptionPayoff(S[i], K) : putOptionPayoff(S[i], K);
            if (currentPayoff > 0) {
                X.push_back(S[i]);
                Y.push_back(payoff[i] * std::exp(-r * dt * (numSteps - j)));
            }
        }
        
        // Simple linear regression (you might want to use a more sophisticated method)
        double sumX = 0, sumY = 0, sumXY = 0, sumX2 = 0;
        int n = X.size();
        for (int i = 0; i < n; ++i) {
            sumX += X[i];
            sumY += Y[i];
            sumXY += X[i] * Y[i];
            sumX2 += X[i] * X[i];
        }
        double beta = (n * sumXY - sumX * sumY) / (n * sumX2 - sumX * sumX);
        double alpha = (sumY - beta * sumX) / n;
        
        for (int i = 0; i < numSimulations; ++i) {
            double continuationValue = alpha + beta * S[i];
            double currentPayoff = isCall ? callOptionPayoff(S[i], K) : putOptionPayoff(S[i], K);
            if (currentPayoff > continuationValue) {
                payoff[i] = currentPayoff;
            } else {
                payoff[i] *= std::exp(-r * dt);
            }
        }
    }
    
    // Calculate the average payoff
    double sumPayoff = 0;
    for (double p : payoff) {
        sumPayoff += p;
    }
    return sumPayoff / numSimulations;
}

int main() {
    // Option parameters
    double S0 = 100.0;   // Initial stock price
    double K = 100.0;    // Strike price
    double r = 0.05;     // Risk-free rate
    double sigma = 0.2;  // Volatility
    double T = 1.0;      // Time to maturity (1 year)
    int numSimulations = 100000; // Number of simulations
    int numSteps = 50;   // Number of time steps for American option

    // Calculate European option prices
    double bsCallPrice = blackScholesPrice(S0, K, r, sigma, T, true);
    double bsPutPrice = blackScholesPrice(S0, K, r, sigma, T, false);
    double mcCallPrice = monteCarloOptionPricing(S0, K, r, sigma, T, numSimulations, true);
    double mcPutPrice = monteCarloOptionPricing(S0, K, r, sigma, T, numSimulations, false);

    // Calculate American option prices
    double amCallPrice = americanOptionPricing(S0, K, r, sigma, T, numSimulations, numSteps, true);
    double amPutPrice = americanOptionPricing(S0, K, r, sigma, T, numSimulations, numSteps, false);

    // Calculate Greeks
    Greeks callGreeks = calculateGreeks(S0, K, r, sigma, T, true);
    Greeks putGreeks = calculateGreeks(S0, K, r, sigma, T, false);

    // Output results
    std::cout << "European Call Option Prices:" << std::endl;
    std::cout << "  Black-Scholes: " << bsCallPrice << std::endl;
    std::cout << "  Monte Carlo: " << mcCallPrice << std::endl;
    std::cout << "European Put Option Prices:" << std::endl;
    std::cout << "  Black-Scholes: " << bsPutPrice << std::endl;
    std::cout << "  Monte Carlo: " << mcPutPrice << std::endl;
    std::cout << "American Option Prices:" << std::endl;
    std::cout << "  Call: " << amCallPrice << std::endl;
    std::cout << "  Put: " << amPutPrice << std::endl;
    
    std::cout << "\nCall Option Greeks:" << std::endl;
    std::cout << "  Delta: " << callGreeks.delta << std::endl;
    std::cout << "  Gamma: " << callGreeks.gamma << std::endl;
    std::cout << "  Theta: " << callGreeks.theta << std::endl;
    std::cout << "  Vega: " << callGreeks.vega << std::endl;
    std::cout << "  Rho: " << callGreeks.rho << std::endl;
    
    std::cout << "\nPut Option Greeks:" << std::endl;
    std::cout << "  Delta: " << putGreeks.delta << std::endl;
    std::cout << "  Gamma: " << putGreeks.gamma << std::endl;
    std::cout << "  Theta: " << putGreeks.theta << std::endl;
    std::cout << "  Vega: " << putGreeks.vega << std::endl;
    std::cout << "  Rho: " << putGreeks.rho << std::endl;

    return 0;
}
