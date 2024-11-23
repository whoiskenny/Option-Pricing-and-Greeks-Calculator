# Option Pricing and Greeks Calculator

This project implements a comprehensive set of tools for pricing financial options using both analytical and numerical methods, alongside calculating key risk metrics (Greeks). It includes implementations for European and American options pricing and supports the calculation of Greeks using the Black-Scholes model and Monte Carlo simulation.

## Features

1. **European Option Pricing**:
   - Black-Scholes analytical pricing model.
   - Monte Carlo simulation for numerical pricing.

2. **American Option Pricing**:
   - Longstaff-Schwartz algorithm for handling early exercise options.

3. **Greeks Calculation**:
   - Delta, Gamma, Theta, Vega, and Rho using the finite difference method.

4. **Monte Carlo Simulations**:
   - Utilizes random paths for robust option price estimation.

5. **Flexible Input Parameters**:
   - Customize stock price, strike price, volatility, risk-free rate, time to maturity, and the number of simulations.

## Key Concepts

- **Black-Scholes Model**: A closed-form solution for pricing European call and put options.
- **Monte Carlo Simulation**: A stochastic approach for estimating option prices by simulating potential future asset paths.
- **Longstaff-Schwartz Algorithm**: A method for pricing American options by approximating the continuation value using regression.

## Usage

- Configure option parameters directly in the source code (`main()` function).
- Outputs include:
  - Prices for European and American options.
  - Risk metrics (Greeks) for European options.

## Prerequisites

- C++ compiler supporting C++11 or later.
- Standard libraries for random number generation and mathematical computations.

## How to Run

1. Clone the repository:
   ```
   git clone https://github.com/whoiskenny/Option-Pricing-and-Greeks-Calculator
   ```
2. Compile the program:
   ```
   g++ -o option_pricing mnto_carlo.cpp -std=c++11
   ```
3. Execute the compiled binary:
   ```
   ./option_pricing
   ```

## Future Enhancements

- Support for exotic options.
- Advanced regression techniques in the Longstaff-Schwartz algorithm.
- Parallelization for Monte Carlo simulations.

This project serves as a practical reference for financial engineering students and professionals interested in quantitative finance and derivatives pricing.
