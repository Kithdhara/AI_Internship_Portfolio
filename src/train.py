import pandas as pd
import joblib  # To save the model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os

def train():
    # 1. Load Data
    data_path = os.path.join("data", "insurance_cleaned.csv")
    df = pd.read_csv(data_path)

    # 2. Separate Inputs (X) and Output (y)
    y = df['charges']
    X = df.drop(columns=['charges'])

    # 3. Define the "Rules" for Columns
    # Numerical columns (Age, BMI, Children) -> Standardize them (Scale)
    # Categorical columns (Sex, Smoker, Region) -> Turn into numbers (OneHot)
    numeric_features = ['age', 'bmi', 'children']
    categorical_features = ['sex', 'smoker', 'region']

    # Create the "Transformer"
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

    # 4. Create the Pipeline (The Magic Step)
    # Step 1: Preprocess data
    # Step 2: Train Linear Regression
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', LinearRegression())
    ])

    # 5. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 6. Train the WHOLE Pipeline
    print("🤖 Training the Pipeline...")
    model_pipeline.fit(X_train, y_train)

    # 7. Evaluate
    predictions = model_pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print("\n--- 📝 Pipeline Report ---")
    print(f"💰 Average Error (MAE): ${mae:,.2f}")
    print(f"📈 Accuracy Score (R2): {r2:.2f}")

    # 8. Save the Model (Crucial for Week 11)
    # We create a 'models' folder first
    if not os.path.exists("models"):
        os.makedirs("models")
        
    joblib.dump(model_pipeline, 'models/insurance_model.pkl')
    print("\n💾 Model saved to 'models/insurance_model.pkl'")

if __name__ == "__main__":
    train()