# Development Steps - Data Center Energy Optimization System

This document outlines the complete development process for the AI-powered data center energy optimization system.

## Overview

The project was built in 5 major steps, each creating a complete module that builds upon the previous one.

---

## Step 1: Dataset Generation (step1_dataset.py)

### Objective
Create a synthetic dataset simulating data center operations.

### Implementation
- **Library Used**: pandas, numpy
- **Records Created**: 500 hourly samples
- **Time Range**: Starting January 1, 2023
- **Features Generated**:
  - `Time`: Hourly timestamps
  - `Energy_Consumption_kWh`: Random values between 300-500 kWh
  - `Server_Load_percent`: Random values between 50-100%
  - `Ambient_Temperature_C`: Random values between 18-30°C
  - `Cooling_Efficiency_percent`: Random values between 70-95%

### Output
- File: `datacenter_data.csv`
- Shape: (500, 5)

### Key Code Highlights
```python
np.random.seed(42)  # Reproducibility
timestamps = [start_date + timedelta(hours=i) for i in range(500)]
df = pd.DataFrame(data)
df.to_csv('datacenter_data.csv', index=False)
```

---

## Step 2: Data Cleaning & Preprocessing (step2_clean.py)

### Objective
Clean the raw data and engineer features for machine learning.

### Implementation
- **Missing Values**: Filled with median values
- **Outlier Removal**: Z-score threshold > 3
- **Feature Engineering**:
  - `Hour`: Extracted from timestamp (0-23)
  - `Day_of_Week`: Extracted from timestamp (0-6)
  - `Month`: Extracted from timestamp (1-12)
  - `Season`: Derived from month (Winter/Spring/Summer/Fall)

### Output
- File: `cleaned_data.csv`
- Shape: (500, 9) - Added 4 new features
- Rows Removed: 0 (no outliers detected)

### Key Code Highlights
```python
df['Hour'] = df['Time'].dt.hour
df['Day_of_Week'] = df['Time'].dt.dayofweek
df['Month'] = df['Time'].dt.month
df['Season'] = df['Month'].apply(get_season)
```

---

## Step 3: Machine Learning Models (step3_models.py)

### Objective
Train and evaluate multiple ML models to predict energy consumption.

### Implementation
- **Features (X)**: Server_Load_percent, Ambient_Temperature_C, Cooling_Efficiency_percent, Hour, Day_of_Week, Month
- **Target (y)**: Energy_Consumption_kWh
- **Train/Test Split**: 80/20
- **Scaling**: StandardScaler for feature normalization

### Models Trained
1. **Linear Regression**
   - MAE: 51.39 kWh
   - RMSE: 59.46 kWh

2. **Random Forest** ⭐ (BEST)
   - MAE: 51.20 kWh
   - RMSE: 61.00 kWh
   - Hyperparameters: n_estimators=100

3. **XGBoost**
   - MAE: 57.83 kWh
   - RMSE: 69.59 kWh

### Output
- File: `best_model.pkl` (Random Forest model)
- File: `scaler.pkl` (StandardScaler)

### Model Selection
Random Forest was selected as the best model based on lowest MAE.

---

## Step 4: Recommendation System (step4_recommend.py)

### Objective
Create an intelligent recommendation engine based on predictions.

### Implementation
- **Function**: `recommend_actions(predicted_energy, server_load, cooling_efficiency)`
- **Recommendation Logic**:

  **Energy-Based**:
  - Energy > 450 kWh: "HIGH - Reduce server load 15-20%"
  - Energy > 420 kWh: "ELEVATED - Monitor cooling"
  - Energy ≤ 420 kWh: "OPTIMAL - Maintain settings"

  **System-Based**:
  - Cooling Efficiency < 75%: "Improve cooling efficiency"
  - Server Load > 80%: "Reduce server load"

### Testing
Tested with 5 different scenarios covering:
- High load + low cooling
- Low load + high cooling
- Mixed conditions

### Output
- Validated recommendation logic
- Tested prediction pipeline

---

## Step 5: Interactive Dashboard (dashboard.py)

### Objective
Build a user-friendly web interface for real-time predictions and visualization.

### Implementation
- **Framework**: Streamlit
- **Visualization**: Plotly

### Dashboard Sections

#### 1. Historical Data Analysis
- Energy Consumption Over Time (line chart)
- Server Load Distribution (histogram)
- Ambient Temperature Distribution (histogram)
- Cooling Efficiency Distribution (histogram)

#### 2. Real-Time Prediction
- **Interactive Sliders**:
  - Server Load: 50-100%
  - Ambient Temperature: 18-30°C
  - Cooling Efficiency: 70-95%
  - Hour of Day: 0-23
  - Day of Week: Monday-Sunday
  - Month: January-December

- **Live Prediction Display**:
  - Predicted energy consumption
  - Color-coded status (Green/Yellow/Red)
  - AI-generated recommendations

#### 3. Key Metrics
- Average Energy Consumption
- Peak Energy Consumption
- Minimum Energy Consumption
- Average Cooling Efficiency

#### 4. Model Information
- Model type and architecture
- Performance metrics (MAE, RMSE)
- Training dataset details

### Features
- Responsive layout with columns
- Real-time updates on slider change
- Color-coded alerts (success/warning/error)
- Professional UI design

---

## Technology Stack

### Core Libraries
- **Data Processing**: pandas 2.3.3, numpy 2.3.5
- **Machine Learning**: scikit-learn 1.8.0, xgboost 3.1.2
- **Statistical Analysis**: scipy 1.16.3
- **Web Framework**: streamlit 1.52.1
- **Visualization**: plotly 6.5.0

### Development Tools
- Python 3.11
- Git for version control
- GitHub for repository hosting

---

## Results Summary

### Model Performance
- **Best Model**: Random Forest Regressor
- **MAE**: 51.20 kWh (±12.8% error on avg 400 kWh)
- **RMSE**: 61.00 kWh
- **Features**: 6 input features
- **Training Data**: 400 samples (80% of 500)
- **Test Data**: 100 samples (20% of 500)

### System Capabilities
- Real-time energy prediction
- Multi-factor analysis
- Intelligent recommendations
- Interactive visualizations
- User-friendly interface

---

## Future Enhancements

1. **Model Improvements**:
   - Collect real-world data
   - Add more features (CPU usage, network traffic)
   - Implement time-series models (LSTM, Prophet)

2. **Dashboard Features**:
   - Historical comparison charts
   - What-if scenario analysis
   - Export reports (PDF/Excel)
   - Email alerts for high consumption

3. **Deployment**:
   - Streamlit Cloud deployment
   - API endpoint creation
   - Database integration (PostgreSQL)
   - User authentication

4. **Optimization**:
   - Automated retraining pipeline
   - A/B testing for recommendations
   - Cost analysis integration

---

## Conclusion

This project successfully demonstrates an end-to-end AI pipeline:
1. Data generation and preprocessing
2. Model training and evaluation
3. Intelligent recommendation system
4. Interactive web dashboard

The system provides actionable insights for data center energy optimization through machine learning predictions and real-time analysis.
