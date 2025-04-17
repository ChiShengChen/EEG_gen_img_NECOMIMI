import numpy as np
from scipy.stats import t
import pandas as pd

# Define accuracy values from the table (N=10 subjects)
data = {
    'BraVL': {
        'top-1': [ 6.1,  4.9,  5.6,  5.0,  4.0,  6.0,  6.5,  8.8,  4.3,  7.0],
        'top-5': [17.9, 14.9, 17.4, 15.1, 13.4, 18.2, 20.4, 23.7, 14.0, 19.7],
    },
    'Nervformer': {
        'top-1': [15.0, 15.6, 19.7, 23.3, 13.0, 18.9, 19.5, 30.3, 20.1, 22.9],
        'top-5': [36.7, 40.0, 44.9, 54.4, 29.1, 42.2, 42.0, 60.0, 46.3, 47.1],
    },
    'NICE': {
        'top-1': [19.3, 15.2, 23.9, 24.1, 11.0, 18.5, 21.0, 32.5, 18.2, 27.4],
        'top-5': [44.8, 38.2, 51.4, 51.6, 30.7, 43.8, 47.9, 63.5, 42.4, 57.1],
    },
    'MUSE': {
        'top-1': [19.8, 15.3, 24.7, 24.7, 12.1, 22.1, 21.0, 33.2, 19.1, 25.0],
        'top-5': [41.1, 34.2, 52.6, 52.6, 33.7, 51.9, 48.6, 59.9, 43.0, 55.2],
    },
    'ATM-S': {
        'top-1': [25.8, 24.6, 28.4, 25.9, 16.2, 21.2, 25.9, 37.9, 26.0, 30.0],
        'top-5': [54.1, 52.6, 62.9, 57.8, 41.9, 53.0, 57.2, 71.1, 53.9, 60.9],
    },
    'NERV (ours)': {
        'top-1': [25.4, 24.1, 28.6, 30.0, 19.3, 24.9, 26.1, 40.8, 27.0, 32.3],
        'top-5': [51.2, 51.1, 53.9, 58.4, 43.9, 52.3, 51.6, 67.4, 55.2, 61.6],
    }
}

# Function to compute 95% CI
def compute_ci(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    t_crit = t.ppf(0.975, df=n-1)
    margin = t_crit * std / np.sqrt(n)
    return mean, margin

# Build results
results = []
for model, values in data.items():
    row = {"Model": model}
    for setting in ['top-1', 'top-5']:
        mean, margin = compute_ci(values[setting])
        row[setting] = f"{mean:.1f} ± {margin:.1f}"
    results.append(row)

df = pd.DataFrame(results)
print(df)
