from autogluon.tabular import TabularPredictor
import pandas as pd

# 1. Load Data
train_data = pd.read_csv("train_data.csv")
val_data = pd.read_csv("val_data.csv")

# 2. Setup Predictor
save_path = 'ag_models_lsd'
predictor = TabularPredictor(label='log_score', path=save_path, problem_type='regression')

# 3. Train
# time_limit=180s (3 min) 
print("Starting training (Limit: 180s)...")
predictor.fit(
    train_data,
    tuning_data=val_data,
    time_limit=180,
    presets='medium_quality'
)

# 4. Evaluate on Validation Data (Leaderboard)
print("\n--- Leaderboard ---")
predictor.leaderboard(val_data, silent=False)
