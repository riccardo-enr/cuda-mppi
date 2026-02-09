## 2024-10-25 - Pandas vs Numpy Performance
**Learning:** Performing squared distance comparisons (`dist_sq < r**2`) on NumPy arrays (`series.values`) is significantly faster (~4x) than using `np.sqrt` on Pandas Series for this codebase's datasets.
**Action:** When filtering DataFrames based on geometric conditions, prefer extracting `.values` and using squared distance comparisons over `np.sqrt` on Series.
