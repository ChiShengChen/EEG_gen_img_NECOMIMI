# Re-import required modules after code state reset
import numpy as np
from scipy.stats import t
import pandas as pd

# Data extracted from image for each model/stage
data_ci = {
    'Nervformer One-Stage': [240.6, 241.7, 259.9, 247.4, 251.5, 241.2, 241.3, 247.1, 250.6, 244.8],
    'Nervformer Two-Stage': [198.3, 201.4, 193.9, 192.8, 206.4, 202.2, 198.5, 202.2, 191.9, 210.8],
    'NICE One-Stage': [238.1, 235.3, 242.3, 241.5, 248.4, 243.5, 224.6, 262.4, 243.6, 237.2],
    'NICE Two-Stage': [204.3, 193.2, 264.9, 198.9, 199.1, 272.0, 235.3, 209.7, 213.1, 196.1],
    'MUSE One-Stage': [231.9, 246.6, 247.0, 257.7, 236.2, 243.1, 243.3, 250.8, 243.0, 245.0],
    'MUSE Two-Stage': [188.7, 218.5, 184.5, 198.1, 215.9, 186.5, 206.7, 190.5, 195.4, 204.7],
    'ATM-S One-Stage': [244.5, 239.7, 238.6, 242.1, 248.9, 250.1, 232.2, 250.3, 239.0, 235.0],
    'ATM-S Two-Stage': [174.9, 194.9, 191.3, 183.0, 194.9, 202.7, 201.5, 179.4, 242.4, 206.1],
    'NERV One-Stage': [217.9, 211.4, 225.6, 221.1, 229.1, 218.8, 215.5, 227.6, 204.6, 217.2],
    'NERV Two-Stage': [169.5, 189.4, 167.2, 175.8, 191.6, 198.3, 187.2, 176.2, 208.1, 174.4],
}

# Compute mean and 95% CI
def compute_ci(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    t_crit = t.ppf(0.975, df=n-1)
    margin = t_crit * std / np.sqrt(n)
    return f"{mean:.2f} ± {margin:.2f}"

ci_results = {
    "Method": [],
    "Mean ± 95% CI": []
}

for method, values in data_ci.items():
    ci_results["Method"].append(method)
    ci_results["Mean ± 95% CI"].append(compute_ci(values))

df_ci = pd.DataFrame(ci_results)
print(df_ci)