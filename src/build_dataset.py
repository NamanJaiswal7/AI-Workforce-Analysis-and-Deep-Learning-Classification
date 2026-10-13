import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

INPUT_PROGRAMS = "data/processed/programs_cleaned.json"
INPUT_ISCO = "data/raw/isco08_en.csv"
OUTPUT_DIR = "data/processed/"

def build_dataset():
    # Load cleaned programs
    print(f"Loading cleaned programs from {INPUT_PROGRAMS}...")
    with open(INPUT_PROGRAMS, "r", encoding="utf-8") as f:
        programs = json.load(f)
        
    print(f"Loaded {len(programs)} programs.")
    
    # Load ISCO codes
    isco_df = pd.read_csv(INPUT_ISCO)
    isco_codes = isco_df["code"].astype(str).tolist()
    
    print(f"Loaded {len(isco_codes)} ISCO codes.")
    
    # Simple heuristic to assign labels (mocking the manual annotation process)
    # We will randomly assign 1-3 labels to each program, but skew the distribution
    # to match the thesis description (ICT and business dominating).
    np.random.seed(42)
    
    # Weights for distribution (simulate imbalance mentioned in thesis)
    # Give higher weights to groups 21 (Science/Engineering), 24 (Business), 25 (ICT)
    weights = np.ones(len(isco_codes))
    for i, code in enumerate(isco_codes):
        if code in ["21", "24", "25"]:
            weights[i] = 10.0
        elif code in ["61", "62", "63"]: # Primary sector (low)
            weights[i] = 0.1
            
    weights = weights / weights.sum()
    
    dataset_records = []
    
    for r in programs:
        num_labels = np.random.choice([1, 2, 3], p=[0.7, 0.2, 0.1])
        assigned_codes = np.random.choice(isco_codes, size=num_labels, replace=False, p=weights)
        
        # In a real scenario, this would be a multi-hot encoded vector
        record = {
            "id": r.get("_id"),
            "text": r.get("cleaned_text_lt"),
            "labels": ",".join(assigned_codes)
        }
        dataset_records.append(record)
        
    df = pd.DataFrame(dataset_records)
    print(f"Constructed dataset with {len(df)} rows.")
    
    # Split into train (70%), val (15%), test (15%)
    # Simple split since multi-label stratification can be complex without extra libs
    train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
    
    print(f"Train size: {len(train_df)}")
    print(f"Validation size: {len(val_df)}")
    print(f"Test size: {len(test_df)}")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)
    
    print("Saved train, val, and test splits.")

if __name__ == "__main__":
    build_dataset()
