import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
import os

def train():
    # 1. Load the CLEAN data
    data_path = os.path.join("data", "insurance_cleaned.csv")
    if not os.path.exists(data_path):
        print("❌ Error: 'insurance_cleaned.csv' not found.")
        return

    df = pd.read_csv(data_path)
    print(f"✅ Data Loaded. Shape: {df.shape}")

    # 2. The "Translator" (Encoding)
    # Computers can't understand "yes"/"no" or "female". We must convert them to numbers.
    # This function turns 'sex', 'smoker', 'region' into 0s and 1s.
    df = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    # 3. Define Features (X) and Target (y)
    # y = What we want to predict (Charges)
    # X = Everything else (Age, BMI, Smoker_Yes, etc.)
    y = df['charges']
    X = df.drop(columns=['charges'])

    # 4. The "Exam" Split (Train/Test Split)
    # We hide 20% of data (test_size=0.2) to test the model later.
    # random_state=42 ensures we get the same shuffle every time (for reproducibility).
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"📊 Training on {len(X_train)} rows. Testing on {len(X_test)} rows.")

    # 5. Train the Model (The "Fit" step)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print("🤖 Model has been trained!")

    # 6. Evaluate (The Grade)
    # Ask the model to predict the costs for the Test set (the data it hasn't seen)
    predictions = model.predict(X_test)
    
    # Calculate the error
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- 📝 Report Card ---")
    print(f"💰 Average Error (MAE): ${mae:,.2f}")
    print(f"📈 Accuracy Score (R2): {r2:.2f} (1.0 is perfect)")

if __name__ == "__main__":
    train()