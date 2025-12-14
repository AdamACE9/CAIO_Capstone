# Data Center Energy Optimization System

An AI-powered data center energy optimization system that uses machine learning to predict energy consumption and provide real-time recommendations.

## Features

- **Machine Learning Models**: Trained Random Forest model for energy prediction (MAE: 51.20 kWh)
- **Real-Time Predictions**: Interactive dashboard with live energy consumption forecasts
- **Smart Recommendations**: Intelligent alerts based on energy levels, cooling efficiency, and server load
- **Interactive Visualizations**: Historical data analysis with Plotly charts
- **User-Friendly Interface**: Streamlit-based dashboard with intuitive controls

## Tech Stack

- **Python 3.11**
- **Machine Learning**: scikit-learn, XGBoost
- **Data Processing**: Pandas, NumPy, SciPy
- **Visualization**: Plotly, Streamlit
- **Model**: Random Forest Regressor

## Live Demo

[Streamlit Cloud Deployment](https://placeholder-link.streamlit.app) *(Coming Soon)*

## Quick Start

See [RUN.md](RUN.md) for detailed instructions on running the project locally.

## Project Structure

```
├── step1_dataset.py          # Dataset generation
├── step2_clean.py            # Data cleaning & preprocessing
├── step3_models.py           # ML model training
├── step4_recommend.py        # Recommendation system
├── dashboard.py              # Streamlit dashboard
├── datacenter_data.csv       # Raw dataset
├── cleaned_data.csv          # Processed dataset
├── best_model.pkl            # Trained ML model
├── scaler.pkl                # Feature scaler
└── requirements.txt          # Python dependencies
```

## Model Performance

- **Algorithm**: Random Forest Regressor
- **Mean Absolute Error (MAE)**: 51.20 kWh
- **Root Mean Squared Error (RMSE)**: 61.00 kWh
- **Features**: 6 (Server Load, Ambient Temperature, Cooling Efficiency, Hour, Day of Week, Month)
- **Training Samples**: 500

## Development Process

See [STEPS.md](STEPS.md) for a detailed breakdown of the development steps.

## License

MIT License

## Author

Adam Ahmed Danish
