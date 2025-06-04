# ⚽ Predicting Big Chances in Football Matches with XGBoost

This is an interactive Streamlit app that uses **StatsBomb Open Data** and machine learning to **predict big chance creation/concession in football matches**. It combines tactical event visualizations, feature engineering, and XGBoost models to provide analysts with both **visual** and **predictive insights** into game momentum and attacking patterns.

---

## 📦 App Features

### ✅ Select & Explore:
- Choose any competition and team from StatsBomb Open Data.
- Browse full match data with goals, shots, carries, and passes.
- Filter and highlight matches using time-binned views.

### 📈 Visual Analytics:
- **Time-binned histograms** and **rolling averages** of:
  - Expected Goals (xG)
  - Final Third Passes
  - Box Passes
  - Final Third Carries
  - Box Carries
- **Shot maps**, **pass maps**, and **carry maps** for selected matches.
- **Time interval filter** to focus on specific 10-minute chunks.

### 🤖 Predictive Modeling:
- Train XGBoost classifiers to forecast:
  - Whether a big chance will occur in the **next 10 minutes**.
  - Separate models for chances **for** and **against**.
- Models use features from previous bins to make predictions.
- Full test/train evaluation with per-match summaries and accuracy metrics.
- Interactive Gantt chart visualizations of predictions vs actual events.

---
## 📊 How It Works

Matches are broken into **10-minute time bins**, and we:
1. **Recalculate match time** so that it runs from 0 to 100 instead of two halves of 0–45+ (1st half) and 45–90+ (2nd half).  
   This allows for consistent binning across the full match and simplifies temporal modeling.
2. **Aggregate binned and rolling stats** for each team.
3. **Label each bin**: does a big chance occur in the next one?
4. **Expand each match** into 9 training rows (bins 0–8).
5. **Train classifiers** to predict future big chances using historical bin features.

### Labels: What is a Big Chance?
A **big chance** is defined as an event (e.g. shot) in a future bin with **xG ≥ your selected threshold**. You can set this value using the slider in the app.

---

## 🧬 Data Source

This app uses data from [StatsBomb Open Data](https://github.com/statsbomb/open-data), specifically:
- Match events (passes, shots, carries, etc.)
- Match metadata (teams, competitions, seasons)
- JSON parsing is handled via custom utilities in `utils/`.

---

## 🗂 Project Structure
📁 modules/
│   ├── visuals.py             # Plots for shots, passes, carries, and prediction timelines
│   └── machine_learning.py    # Feature engineering, model training, and evaluation pipeline

📁 utils/
│   ├── data_fetcher.py        # Functions for loading, processing, and caching StatsBomb JSON data
│   └── page_components.py     # Streamlit sidebar layout and shared UI components

📄 main.py                     # Entry point for the Streamlit app
📄 requirements.txt            # Python dependencies for setup

---

## 🙌 Acknowledgements
💾 StatsBomb for their generous open data.

📚 Streamlit for making interactive ML tools simple.

🧠 Built as a personal project to combine tactical football intelligence with machine learning.

## ⚠️ Disclaimer

This tool is for educational and demonstrative purposes only. It is not affiliated with StatsBomb or any football clubs.



