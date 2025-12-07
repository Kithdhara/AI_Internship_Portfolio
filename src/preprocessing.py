import pandas as pd
import os

def load_data(filepath):
    """Loads data from a CSV file."""
    if not os.path.exists(filepath):
        print(f"Error: The file '{filepath}' was not found.")
        return None
    try:
        df = pd.read_csv(filepath)
        print(f"✅ Success! Data loaded from {filepath}")
        print(f"   Original Shape: {df.shape}")
        return df
    except Exception as e:
        print(f"Error loading file: {e}")
        return None

def clean_data(df):
    """Removes duplicates and handles missing values."""
    # 1. Check for duplicates
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"⚠️ Found {duplicates} duplicate rows. Removing them...")
        df = df.drop_duplicates()
    
    # 2. Check for missing values
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f"⚠️ Found {missing_values} missing values. Filling with 0...")
        df = df.fillna(0)
    else:
        print("✅ No missing values found.")

    print(f"   Final Shape: {df.shape}")
    return df

if __name__ == "__main__":
    DATA_PATH = os.path.join("data", "insurance.csv")
    
    # 1. Load
    df = load_data(DATA_PATH)
    
    # 2. Clean (Only if load was successful)
    if df is not None:
        df_clean = clean_data(df)
        
        # 3. Save the clean version
        df_clean.to_csv(os.path.join("data", "insurance_cleaned.csv"), index=False)
        print("💾 Cleaned data saved to 'data/insurance_cleaned.csv'")