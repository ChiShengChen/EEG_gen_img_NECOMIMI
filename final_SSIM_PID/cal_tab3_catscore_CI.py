import numpy as np
from scipy.stats import t
import pandas as pd

# Define accuracy values from the table (N=10 subjects)
data = {
    'Nervformer': {
        'catscore': [432, 457, 429, 454, 475, 463, 404, 438, 427, 410],
    },
    'NICE': {
        'catscore': [426, 456, 445, 447, 411, 454, 438, 443, 426, 429],
    },
    'MUSE': {
        'catscore': [438, 456, 434, 416, 426, 463, 443, 437, 420, 468],
    },
    'ATM-S': {
        'catscore': [413, 419, 411, 464, 427, 469, 442, 472, 431, 445],
    },
    'NERV (ours)': {
        'catscore': [445, 436, 432, 456, 438, 466, 410, 437, 433, 444],
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
    for setting in ['catscore']:
        mean, margin = compute_ci(values[setting])
        row[setting] = f"{mean:.1f} ± {margin:.1f}"
    results.append(row)

df = pd.DataFrame(results)
print(df)
