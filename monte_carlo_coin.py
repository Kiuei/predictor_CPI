import numpy as np
import matplotlib.pyplot as plt

# Parameters
initial_money = 100
bets = 10
simulations = 1000

# Run Monte Carlo Simulation
final_balances = []

for _ in range(simulations):
    money = initial_money
    for _ in range(bets):
        if np.random.rand() < 0.5:  # 50% chance for heads
            money += 10
        else:
            money -= 5
    final_balances.append(money)

# Plot Results
plt.hist(final_balances, bins=20, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(final_balances), color='r', linestyle='dashed', linewidth=2, label=f'Avg: ${np.mean(final_balances):.2f}')
plt.title("Monte Carlo Simulation: Coin Toss Betting")
plt.xlabel("Final Money ($)")
plt.ylabel("Frequency")
plt.legend()
plt.show()
