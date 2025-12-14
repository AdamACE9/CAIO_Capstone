# How to Run the Data Center Energy Optimization System

## Live Demo

Go to https://adam-caio.streamlit.app/ to see it in action!

---

## Or run it locally:

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the dashboard

```bash
streamlit run dashboard.py
```

Or if that doesn't work:

```bash
python -m streamlit run dashboard.py
```

The dashboard will open automatically in your browser at `http://localhost:8501`

---

## Optional: Regenerate Data & Retrain Models

If you want to start fresh:

```bash
python step1_dataset.py
python step2_clean.py
python step3_models.py
python step4_recommend.py
```
