import numpy as np
from scipy.stats import t
import pandas as pd

# Define accuracy values from the table (N=10 subjects)
data = {
    'Nervformer': {
        '2-way': [89.9, 91.3, 91.6, 94.3, 86.3, 91.1, 92.5, 96.2, 92.0, 92.4],
        '4-way': [76.9, 80.7, 80.8, 85.9, 70.4, 82.5, 81.6, 88.3, 83.7, 83.1],
    },
    'NICE': {
        '2-way': [91.7, 89.8, 93.5, 94.0, 85.9, 89.1, 91.2, 95.8, 87.9, 93.8],
        '4-way': [80.4, 77.4, 83.7, 84.9, 70.3, 81.7, 81.7, 89.2, 76.5, 87.1],
    },
    'MUSE': {
        '2-way': [90.1, 90.3, 93.4, 93.6, 88.3, 93.1, 93.1, 95.4, 90.5, 94.4],
        '4-way': [78.4, 76.8, 85.6, 87.5, 74.2, 85.3, 82.8, 87.7, 81.8, 88.1],
    },
    'ATM-S': {
        '2-way': [94.8, 93.5, 95.3, 95.9, 90.8, 94.1, 94.2, 96.6, 94.1, 94.7],
        '4-way': [84.9, 86.3, 89.0, 87.3, 87.5, 85.2, 87.1, 92.9, 86.8, 87.0],
    },
    'NERV (ours)': {
        '2-way': [95.3, 96.0, 95.9, 95.8, 90.8, 93.6, 94.7, 96.8, 94.4, 94.8],
        '4-way': [85.7, 88.8, 91.2, 87.4, 80.4, 84.0, 86.2, 92.3, 84.2, 87.6],
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
    for setting in ['2-way', '4-way']:
        mean, margin = compute_ci(values[setting])
        row[setting] = f"{mean:.1f} ± {margin:.1f}"
    results.append(row)

df = pd.DataFrame(results)
print(df)
