import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from molfeat.calc import FPCalculator
from molfeat.trans import MoleculeTransformer

# 1. Load data
print("Loading data...")
df = pd.read_csv("lsd_small_subset.csv")

# 2. Transform Score (Using 'score' column)
# Subtraction of smallest value and log(x+1) as per lab instructions
min_val = df['score'].min()
df['log_score'] = np.log1p(df['score'] - min_val)
print("Scores transformed.")

# 3. Compute Features (ECFP4) using molfeat
print("Computing features...")
calc = FPCalculator("ecfp", radius=2, length=1024)
trans = MoleculeTransformer(calc)

# Convert SMILES to fingerprints
raw_features = trans.transform(df['smiles'].values)

# Convert list to NumPy array so we can access .shape
features = np.array(raw_features)

# Create separate columns for each feature (fp_0, fp_1, etc.)
feat_cols = [f"fp_{i}" for i in range(features.shape[1])]
feat_df = pd.DataFrame(features, columns=feat_cols)

# Combine log_score with features
ml_data = pd.concat([df[['log_score']], feat_df], axis=1)

# 4. Split Data (60% Train, 20% Val, 20% Test)
# Splitting twice
train_df, temp_df = train_test_split(ml_data, train_size=0.6, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# 5. Save Files
train_df.to_csv("train_data.csv", index=False)
val_df.to_csv("val_data.csv", index=False)
test_df.to_csv("test_data.csv", index=False)

print("-" * 30)
print("Files created: train_data.csv, val_data.csv, test_data.csv")
print("-" * 30)
