import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pickle

# Load cleaned data
print("Loading cleaned dataset...")
df = pd.read_csv('cleaned_data.csv')

# Prepare features and target
# Features: All except Energy_Consumption_kWh and Time
feature_cols = [col for col in df.columns if col not in ['Energy_Consumption_kWh', 'Time', 'Season']]
X = df[feature_cols].copy()
y = df['Energy_Consumption_kWh'].copy()

print(f"Features: {feature_cols}")
print(f"Dataset shape: X={X.shape}, y={y.shape}")

# Split data 80/20 train/test
print("\nSplitting data (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train models
print("\n" + "="*60)
print("TRAINING MODELS")
print("="*60)

models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': XGBRegressor(n_estimators=100, random_state=42, verbosity=0)
}

results = {}

for model_name, model in models.items():
    print(f"\nTraining {model_name}...")

    # Train model
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[model_name] = {'model': model, 'mae': mae, 'rmse': rmse}

    print(f"{model_name} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

# Find best model (lowest MAE)
print("\n" + "="*60)
print("MODEL EVALUATION RESULTS")
print("="*60)
best_model_name = min(results, key=lambda x: results[x]['mae'])
best_model = results[best_model_name]['model']

print(f"\nBest Model: {best_model_name}")
print(f"MAE: {results[best_model_name]['mae']:.2f}")
print(f"RMSE: {results[best_model_name]['rmse']:.2f}")

# Save best model and scaler
print("\nSaving best model and scaler...")
with open('best_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("\n[SUCCESS] Model training completed!")
print("Files saved: best_model.pkl, scaler.pkl")
