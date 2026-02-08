import pandas as pd
from datasets import load_dataset

# 1. Define the Hugging Face dataset ID
dataset_id = "IrwinLab/LSD_mPro_Fink_2023_noncovalent_Screen2_ZINC22"

print(f"Streaming dataset: {dataset_id}...")

# 2. Stream the dataset (don't download the whole thing)
# We stream to handle the massive size efficiently
ds = load_dataset(dataset_id, split="train", streaming=True)

# 3. Shuffle and take 50,000 samples
# buffer_size ensures we get a good random mix from the stream
shuffled_ds = ds.shuffle(seed=42, buffer_size=10000)
data_subset = list(shuffled_ds.take(50000))

# 4. Convert to Pandas DataFrame and Save
df = pd.DataFrame(data_subset)
output_file = "lsd_small_subset.csv"
df.to_csv(output_file, index=False)

print(f"Saved {len(df)} rows to {output_file}")
