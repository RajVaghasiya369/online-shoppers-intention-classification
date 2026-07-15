# Online Shoppers Intention Prediction

Machine learning project that predicts whether an online shopping session will generate **Revenue** (a purchase), with a Flask web app for interactive and API-based predictions.

## Project Overview

The pipeline explores session behavior data, engineers features, handles class imbalance with SMOTE, tunes seven classifiers, and persists the best model plus a fitted preprocessor for deployment.

**Best saved model:** AdaBoost  
**Holdout metrics:** accuracy ≈ 0.89 · precision ≈ 0.64 · recall ≈ 0.72 · F1 ≈ 0.68

## Business Problem

Predict purchase intent from:
- Page view patterns (Administrative, Informational, Product)
- Duration metrics
- Bounce and exit rates
- Page values
- Temporal features (Month, Weekend)
- Visitor characteristics (Type, Traffic source, OS, Browser, Region)

## Dataset

**Location:** `data/online_shoppers_intention.csv` (~12,330 rows)

| Feature | Description |
|---------|-------------|
| Administrative | Administrative pages visited |
| Administrative_Duration | Time on administrative pages |
| Informational | Informational pages visited |
| Informational_Duration | Time on informational pages |
| ProductRelated | Product-related pages visited |
| ProductRelated_Duration | Time on product pages |
| BounceRates | Bounce rate |
| ExitRates | Exit rate |
| PageValues | Average page value |
| SpecialDay | Proximity to special days |
| Month | Month of visit |
| OperatingSystems | OS identifier |
| Browser | Browser identifier |
| Region | Geographic region |
| TrafficType | Traffic source type |
| VisitorType | New / Returning / Other |
| Weekend | Weekend visit (boolean) |
| **Revenue** | **Target** — purchase made (True/False) |

## Project Structure

```
online_shoppers_intention/
├── app.py                          # Flask web app + prediction API
├── templates/
│   └── index.html                  # Prediction form UI
├── data/
│   └── online_shoppers_intention.csv
├── src/
│   └── component/
│       ├── classification.py       # Training, EDA, ModelSaver, evaluation
│       └── models/
│           ├── best_models/        # Best model (.joblib) + metadata
│           ├── tuned_models/       # All tuned models + metadata
│           └── preprocessors/      # Fitted ColumnTransformer
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

**Prerequisites:** Python 3.8+

```bash
git clone <repository-url>
cd online_shoppers_intention

python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate

pip install -r requirements.txt
```

### Dependencies

- numpy, pandas — data handling  
- matplotlib, seaborn — visualization  
- scikit-learn — models & preprocessing  
- xgboost — gradient boosting  
- imbalanced-learn — SMOTE  
- flask, joblib — web app & model I/O  

## Train Models

```bash
cd src/component
python classification.py
```

This runs the full pipeline:

1. Load & clean data  
2. Feature engineering (+ log transforms for skewed features)  
3. Stratified train/test split with leakage-safe preprocessing  
4. SMOTE for class imbalance (except XGBoost, which uses `scale_pos_weight`)  
5. RandomizedSearchCV (F1) for 7 models  
6. Evaluation and automatic save of models, best model, metadata, and preprocessor  

Artifacts are written under `src/component/models/`.

## Flask Web App

The app loads the latest best model and preprocessor from `src/component/models/` and serves a form UI plus a JSON API.

### Run the app

From the project root (after models are trained/saved):

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

Optional:

```bash
set PORT=8080          # Windows
export PORT=8080       # Linux / macOS
python app.py
```

### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Interactive prediction form (with model dropdown) |
| `POST` | `/predict` | Predict from form or JSON body (optional `model`) |
| `GET` | `/models` | List available models and the default (best) |
| `GET` | `/health` | Model / preprocessor readiness |

### Model selection

All tuned models under `src/component/models/tuned_models/` are loaded at startup. The best model is used by default.

- **UI:** choose a model from the **Model** dropdown on the form  
- **API:** pass `"model": "RandomForest"` (or any loaded name). Omit to use the best model.

```bash
curl http://127.0.0.1:5000/models
```

### JSON API example

```bash
curl -X POST http://127.0.0.1:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"Administrative\":5,\"Administrative_Duration\":100,\"Informational\":1,\"Informational_Duration\":50,\"ProductRelated\":40,\"ProductRelated_Duration\":1800,\"BounceRates\":0.01,\"ExitRates\":0.02,\"PageValues\":45,\"SpecialDay\":0,\"Month\":\"Nov\",\"OperatingSystems\":2,\"Browser\":2,\"Region\":1,\"TrafficType\":2,\"VisitorType\":\"Returning_Visitor\",\"Weekend\":false}"
```

Linux / macOS:

```bash
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "model": "RandomForest",
    "Administrative": 5,
    "Administrative_Duration": 100,
    "Informational": 1,
    "Informational_Duration": 50,
    "ProductRelated": 40,
    "ProductRelated_Duration": 1800,
    "BounceRates": 0.01,
    "ExitRates": 0.02,
    "PageValues": 45,
    "SpecialDay": 0,
    "Month": "Nov",
    "OperatingSystems": 2,
    "Browser": 2,
    "Region": 1,
    "TrafficType": 2,
    "VisitorType": "Returning_Visitor",
    "Weekend": false
  }'
```

Example response:

```json
{
  "success": true,
  "prediction": 1,
  "revenue": true,
  "label": "Purchase Likely",
  "probability": 0.73,
  "model": "AdaBoost"
}
```

**Notes**
- `Month` values: `Feb`, `Mar`, `May`, `June`, `Jul`, `Aug`, `Sep`, `Oct`, `Nov`, `Dec` (use `June`, not `Jun`)
- `VisitorType`: `Returning_Visitor`, `New_Visitor`, `Other`
- Train models first if `.joblib` files are missing (they are gitignored)

## Feature Engineering

Derived features:
- `TotalPages`, `TotalDuration`
- `ValuePerProduct`, `PageValuePerDuration`
- `ProductFocusRatio`, `ProductTimeRatio`, `BounceExitOff`
- Log transforms (`*_log`) for skewed numeric columns

Before modeling, `Informational`, `SpecialDay`, and `Weekend` are dropped from inputs (engineered / log variants may still be used).

## Models Compared

1. Logistic Regression  
2. Decision Tree  
3. Random Forest  
4. Gradient Boosting  
5. AdaBoost  
6. Bagging Classifier  
7. XGBoost  

Tuning: RandomizedSearchCV, 5-fold CV, F1 scoring.

## Programmatic Training / Loading

```python
from src.component.classification import MyClassificationModel

mcm = MyClassificationModel(
    file_path="data/online_shoppers_intention.csv",
    save_models=True,
)

df = mcm.load_data()
_, _, _, _, df_clean = mcm.missing_duplicate_value(df=df)
df_fe = mcm.feature_enginnering(df=df_clean)
X_train, X_test, y_train, y_test, X, y, preprocessor = mcm.train_test_split(df=df_fe)

# Load saved artifacts for inference
model = mcm.load_saved_model(
    "src/component/models/best_models/best_model_AdaBoost_YYYYMMDD_HHMMSS.joblib"
)
prep = mcm.load_saved_preprocessor(
    "src/component/models/preprocessors/data_preprocessor_YYYYMMDD_HHMMSS.joblib"
)
```

`ModelSaver` lives inside `classification.py` (save/load models, preprocessor, and JSON metadata).

## Key Implementation Details

- **No data leakage:** split first; fit preprocessor only on train  
- **SMOTE** only on training data; XGBoost uses `scale_pos_weight` instead  
- **Persistence:** timestamped `.joblib` models + JSON metadata under `src/component/models/`  
- **Inference:** `app.py` applies the same feature engineering + saved preprocessor before predict  

## License

MIT License

## Acknowledgments

- Dataset: [Online Shoppers Purchasing Intention Dataset](https://archive.ics.uci.edu/dataset/468/online+shoppers+purchasing+intention+dataset)  
- Libraries: scikit-learn, XGBoost, imbalanced-learn, Flask  
