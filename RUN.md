# How to Run the Data Center Energy Optimization System

## Prerequisites

- Python 3.11 or higher
- pip (Python package manager)
- Git (optional, for cloning)

## Installation Steps

### 1. Clone the Repository (if not already done)

```bash
git clone https://github.com/AdamACE9/CAIO_Capstone.git
cd CAIO_Capstone
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- pandas
- numpy
- scikit-learn
- xgboost
- scipy
- streamlit
- plotly

### 3. Generate and Process Data (Optional - Already Done)

If you want to regenerate the data and retrain models:

```bash
# Step 1: Generate synthetic dataset
python step1_dataset.py

# Step 2: Clean and preprocess data
python step2_clean.py

# Step 3: Train machine learning models
python step3_models.py

# Step 4: Test recommendation system
python step4_recommend.py
```

### 4. Run the Dashboard

**Option A: Command Line**
```bash
streamlit run dashboard.py
```

**Option B: Python Module**
```bash
python -m streamlit run dashboard.py
```

**Option C: Windows Batch File**
- Double-click `run_dashboard.bat`

### 5. Access the Dashboard

The dashboard will automatically open in your default browser at:
```
http://localhost:8501
```

If it doesn't open automatically, navigate to the URL above.

## Using the Dashboard

1. **View Historical Data**: Explore energy consumption trends and distributions
2. **Make Predictions**: Adjust sliders for:
   - Server Load (50-100%)
   - Ambient Temperature (18-30°C)
   - Cooling Efficiency (70-95%)
   - Hour of Day (0-23)
   - Day of Week (Monday-Sunday)
   - Month (January-December)
3. **Get Recommendations**: View AI-generated optimization suggestions
4. **Monitor Metrics**: Check average, peak, and minimum energy consumption

## Live Demo

**Streamlit Cloud**: [https://placeholder-link.streamlit.app](https://placeholder-link.streamlit.app)

*Update this link once deployed!*

## Troubleshooting

### Issue: `streamlit` command not found

**Solution**: Use the Python module syntax:
```bash
python -m streamlit run dashboard.py
```

### Issue: Missing dependencies

**Solution**: Reinstall requirements:
```bash
pip install -r requirements.txt --upgrade
```

### Issue: Port already in use

**Solution**: Specify a different port:
```bash
streamlit run dashboard.py --server.port 8502
```

## System Requirements

- **RAM**: Minimum 4GB (8GB recommended)
- **Storage**: ~100MB for dependencies
- **Browser**: Chrome, Firefox, Safari, or Edge (latest versions)

## Project Files

- `datacenter_data.csv` - Raw synthetic dataset (500 records)
- `cleaned_data.csv` - Processed dataset with features
- `best_model.pkl` - Trained Random Forest model
- `scaler.pkl` - StandardScaler for feature normalization

## Support

For issues or questions, please open an issue on the [GitHub repository](https://github.com/AdamACE9/CAIO_Capstone/issues).
