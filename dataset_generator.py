import numpy as np
import pandas as pd


S0_min, S0_max   = 50, 150
K_min, K_max     = 50, 150
sigma_min, sigma_max = 0.1, 0.5
T_min, T_max     = 0.1, 2.0
r_min, r_max = 0.0, 0.08

### Monte-Carlo simulation
def price_asian_mc(S0, K, T, sigma, r, n_paths=10_000, n_steps=100):
    dt = T / n_steps
    S = np.zeros((n_paths, n_steps+1))
    S[:, 0] = S0
    for t in range(1, n_steps+1):
        Z = np.random.randn(n_paths)
        S[:, t] = S[:, t-1] * np.exp((r - 0.5*sigma**2)*dt
                                     + sigma*np.sqrt(dt)*Z)
    A = S.mean(axis=1)
    payoff = np.exp(-r*T) * np.maximum(A - K, 0)
    return payoff.mean()

def generate_dataset(n_samples=100_000):
    np.random.seed(42)
    data = {
        'S0':    np.random.uniform(S0_min, S0_max, n_samples),
        'K':     np.random.uniform(K_min, K_max, n_samples),
        'sigma': np.random.uniform(sigma_min, sigma_max, n_samples),
        'T':     np.random.uniform(T_min, T_max, n_samples),
        'r':     np.random.uniform(r_min, r_max, n_samples),
        'price': np.zeros(n_samples)
    }

    for i in range(n_samples):
        data['price'][i] = price_asian_mc(
            data['S0'][i], data['K'][i],
            data['T'][i], data['sigma'][i],
            data['r'][i],
            n_paths=5_000, n_steps=100
        )
        if i % 5_000 == 0:
            print(f"Simulations : {i}/{n_samples}")

    df = pd.DataFrame(data)
    df.to_csv('data/pricing_dataset.csv', index=False)
    print("Dataset généré : data/pricing_dataset.csv")

if __name__ == '__main__':
    generate_dataset()
