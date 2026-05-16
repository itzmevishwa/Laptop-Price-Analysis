# 💻 Laptop Price Analysis & Prediction

An end-to-end machine learning project that predicts laptop prices based on hardware specifications using Random Forest regression. Includes EDA, feature engineering, model comparison, and a deployed Streamlit web app for live predictions.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Scikit--learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Deployed-red.svg)

---

## 🎯 Project Overview

**Goal:** Predict the price of laptops (in Euros) from their hardware specifications, and identify which features drive laptop prices the most.

**Dataset:** 1,275 laptops with 23 features (brand, RAM, CPU, GPU, storage, screen, etc.)

**Approach:**
1. Exploratory Data Analysis (EDA)
2. Feature engineering (PPI from screen resolution)
3. One-hot encoding for categorical variables
4. Train Linear Regression (baseline) + Random Forest (main model)
5. Model comparison using R², MAE, RMSE
6. Feature importance analysis
7. Deploy as a Streamlit web app

---

## 📊 Results

| Model | R² Score | MAE (€) | RMSE (€) |
|-------|----------|---------|----------|
| Linear Regression | 0.7576 | 252.49 | 346.86 |
| **Random Forest** | **0.8544** | **183.52** | **268.80** |

**Random Forest outperforms Linear Regression by ~10% R².** On average, predictions are within €184 of the actual price.

---

## 🔍 Key Insights

**Top features driving laptop price:**
1. **RAM** (~55% importance) — by far the strongest predictor
2. **Weight** (~10%) — heavier laptops tend to be workstations/gaming
3. **Laptop Type** (Notebook, Workstation, Gaming) — strong categorical signal
4. **CPU Frequency** — faster CPUs = pricier laptops
5. **PPI (Pixels Per Inch)** — engineered feature for screen quality

**Brand insights:**
- Premium: Razer, LG, MSI, Apple, Microsoft
- Mid-range: Dell, HP, Asus, Lenovo
- Budget: Mediacom, Vero, Chuwi

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Pandas, NumPy** — data manipulation
- **Matplotlib, Seaborn** — visualization
- **Scikit-learn** — Linear Regression, Random Forest
- **Streamlit** — web app
- **Joblib** — model persistence

---

## 📁 Project Structure

```
Laptop-Price-Analysis/
├── data/
│   └── laptop_prices.csv              # Raw dataset
├── notebooks/
│   └── 01_EDA_and_Modeling.ipynb      # Full EDA + modeling workflow
├── src/
│   └── preprocessing.py               # Reusable preprocessing pipeline
├── models/
│   ├── random_forest_model.pkl        # Trained Random Forest
│   └── feature_columns.pkl            # Feature column order for predictions
├── app.py                             # Streamlit web app
├── requirements.txt                   # Python dependencies
└── README.md
```

---

## 🚀 How to Run Locally

**1. Clone the repo:**
```bash
git clone https://github.com/itzmevishwa/Laptop-Price-Analysis.git
cd Laptop-Price-Analysis
```

**2. Create a virtual environment:**
```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac/Linux
```

**3. Install dependencies:**
```bash
pip install -r requirements.txt
```

**4. Run the notebook** (to reproduce training):
```bash
jupyter notebook notebooks/01_EDA_and_Modeling.ipynb
```

**5. Launch the Streamlit app:**
```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 🎨 Feature Engineering Highlights

**PPI (Pixels Per Inch):** Instead of using `ScreenW`, `ScreenH`, and `Inches` as three separate features, they were combined into a single PPI feature:

```
PPI = √(ScreenW² + ScreenH²) / Inches
```

This single feature has stronger correlation with price (0.47) than any of the original three columns individually.

---

## 📈 Model Choice Rationale

**Linear Regression** was used as a baseline because it's simple, interpretable, and fast.

**Random Forest** was chosen as the main model because:
- Handles non-linear relationships naturally
- Captures feature interactions (e.g., RAM × LaptopType)
- Robust to outliers and skewed distributions (price is right-skewed)
- Provides built-in feature importance for interpretability

---

## 👤 Author

**Vishwa A**  
B.Tech Computer Science (AI & ML), SRM Institute of Science and Technology  
📧 vishwa.pvt.01@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/vishwa444/) | [GitHub](https://github.com/itzmevishwa)

---

## 📜 License

This project is open source and available under the MIT License.
