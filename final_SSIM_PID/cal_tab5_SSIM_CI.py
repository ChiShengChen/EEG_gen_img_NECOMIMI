# Re-import required modules after code reset
import numpy as np
from scipy.stats import t
import pandas as pd

# SSIM data extracted from the image for each model/stage
ssim_data = {
    'Pure Test Image': [0.2172] * 10,
    'Nervformer One-Stage': [0.1260, 0.1479, 0.1258, 0.1494, 0.1237, 0.1373, 0.1222, 0.1401, 0.1483, 0.1431],
    'Nervformer Two-Stage': [0.1836, 0.1898, 0.1911, 0.1873, 0.1892, 0.1997, 0.1863, 0.1864, 0.2451, 0.1939],
    'NICE One-Stage': [0.1352, 0.1548, 0.1628, 0.1648, 0.1562, 0.1594, 0.1654, 0.1673, 0.1705, 0.1677],
    'NICE Two-Stage': [0.2240, 0.2276, 0.2248, 0.2284, 0.1873, 0.1419, 0.1873, 0.1895, 0.2127, 0.1919],
    'MUSE One-Stage': [0.1917, 0.1496, 0.1662, 0.1732, 0.1667, 0.1723, 0.1849, 0.1631, 0.1752, 0.1747],
    'MUSE Two-Stage': [0.2586, 0.2108, 0.2249, 0.2212, 0.2174, 0.2131, 0.2486, 0.2185, 0.2047, 0.2478],
    'ATM-S One-Stage': [0.1821, 0.1744, 0.1685, 0.1727, 0.1814, 0.1487, 0.1611, 0.1688, 0.1844, 0.1987],
    'ATM-S Two-Stage': [0.2438, 0.2420, 0.2403, 0.2337, 0.2456, 0.1948, 0.2037, 0.2410, 0.2339, 0.2348],
    'NERV One-Stage': [0.1890, 0.1954, 0.1878, 0.1790, 0.1622, 0.1556, 0.1716, 0.1809, 0.2037, 0.1867],
    'NERV Two-Stage': [0.2337, 0.1909, 0.2223, 0.1982, 0.1762, 0.1884, 0.1877, 0.2095, 0.2177, 0.2138],
}

# Compute mean and 95% CI
def compute_ci(values):
    mean = np.mean(values)
    std = np.std(values, ddof=1)
    n = len(values)
    t_crit = t.ppf(0.975, df=n-1)
    margin = t_crit * std / np.sqrt(n)
    return f"{mean:.4f} ± {margin:.4f}"

ci_results_ssim = {
    "Method": [],
    "Mean SSIM ± 95% CI": []
}

for method, values in ssim_data.items():
    ci_results_ssim["Method"].append(method)
    ci_results_ssim["Mean SSIM ± 95% CI"].append(compute_ci(values))

df_ssim_ci = pd.DataFrame(ci_results_ssim)
print(df_ssim_ci)