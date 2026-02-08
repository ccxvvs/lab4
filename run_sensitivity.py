import pandas as pd
import time
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor

# 1. Load Data
full_train_data = pd.read_csv("train_data.csv")
test_data = pd.read_csv("test_data.csv")

# 2. Define Experiments: 10%, 50%, and 100% of the 30k training set
fractions = [0.1, 0.5, 1.0]
results = []

print(f"\n--- Starting Sensitivity Analysis ---")

for frac in fractions:
    subset = full_train_data.sample(frac=frac, random_state=42)
    n_samples = len(subset)
    
    print(f"\nTraining on {int(frac*100)}% data ({n_samples} rows)...")
    start_time = time.time()
    
    # Train small model
    predictor = TabularPredictor(label='log_score', path=f"ag_model_{int(frac*100)}", problem_type='regression').fit(
        subset,
        time_limit=60, # Restricted time limit for experiment
        presets='medium_quality',
        verbosity=1
    )
    
    duration = time.time() - start_time
    perf = predictor.evaluate(test_data)
    
    results.append({
        'Data Size': n_samples,
        'Time (s)': duration,
        'R2': perf['r2']
    })

# 3. Plotting
df = pd.DataFrame(results)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Time vs Data
ax1.plot(df['Data Size'], df['Time (s)'], marker='o', color='orange')
ax1.set_title("Compute: Time vs Data Size")
ax1.set_xlabel("Molecules"); ax1.set_ylabel("Time (s)")

# Accuracy vs Data
ax2.plot(df['Data Size'], df['R2'], marker='o', color='blue')
ax2.set_title("Performance: R2 vs Data Size")
ax2.set_xlabel("Molecules"); ax2.set_ylabel("R2 Score")

plt.tight_layout()
plt.savefig('sensitivity_plots.png')
print("Plots saved to sensitivity_plots.png")
