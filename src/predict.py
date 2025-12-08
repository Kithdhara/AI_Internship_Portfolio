import joblib
import pandas as pd
import os

def make_prediction():
    # 1. Load the saved pipeline
    model_path = os.path.join("models", "insurance_model.pkl")
    if not os.path.exists(model_path):
        print("❌ Model not found! Run train.py first.")
        return

    model = joblib.load(model_path)
    print("✅ Model loaded successfully!")

    # 2. Define a New Person (Raw Data!)
    # Notice: We write "female", "yes". We do NOT write 0 or 1.
    # The pipeline handles the translation.
    new_person = pd.DataFrame({
        'age': [30],
        'sex': ['male'],
        'bmi': [30.5],
        'children': [0],
        'smoker': ['yes'],
        'region': ['southwest']
    })

    print("\n🔮 Predicting for:")
    print(new_person.to_string(index=False))

    # 3. Predict
    prediction = model.predict(new_person)
    print(f"\n💰 Estimated Insurance Cost: ${prediction[0]:,.2f}")

if __name__ == "__main__":
    make_prediction()