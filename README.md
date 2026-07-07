# Online Shoppers Intention Prediction

A comprehensive machine learning project for predicting online shopping purchase intention using multiple classification algorithms with advanced preprocessing and feature engineering.

## 📋 Project Overview

This project analyzes online shopping session data to predict whether a visitor will make a purchase (generate revenue). The implementation includes extensive data exploration, feature engineering, handling class imbalance, and comparison of multiple classification models.

## 🎯 Business Problem

Predict whether an online shopping session will result in a purchase based on visitor behavior metrics such as:
- Page view patterns (Administrative, Informational, Product pages)
- Duration metrics
- Bounce and exit rates
- Page values
- Temporal features (Month, Weekend)
- Visitor characteristics (Type, Traffic source, Operating system, Browser)

## 📊 Dataset

**Location:** `data/online_shoppers_intention.csv`

**Key Features:**
- **Administrative**: Number of administrative pages visited
- **Administrative_Duration**: Time spent on administrative pages
- **Informational**: Number of informational pages visited
- **Informational_Duration**: Time spent on informational pages
- **ProductRelated**: Number of product-related pages visited
- **ProductRelated_Duration**: Time spent on product pages
- **BounceRates**: Percentage of visitors who leave after viewing only one page
- **ExitRates**: Percentage of exits from a specific page
- **PageValues**: Average value of pages visited
- **SpecialDay**: Proximity to special days (e.g., holidays)
- **Month**: Month of the visit
- **OperatingSystems**: Operating system identifier
- **Browser**: Browser identifier
- **Region**: Geographic region
- **TrafficType**: Traffic source type
- **VisitorType**: New, Returning, or Other visitor
- **Weekend**: Boolean indicating weekend visit
- **Revenue**: Target variable (True/False - Purchase made)

## 🚀 Features

### 1. Data Preprocessing
- **Missing value handling**: Removal of null values
- **Duplicate removal**: Eliminates duplicate records
- **Outlier detection**: IQR-based outlier capping with median replacement
- **Encoding strategies**:
  - Ordinal encoding for Month (temporal order preserved)
  - One-hot encoding for VisitorType
  - Binary encoding for boolean features (Weekend, Revenue)

### 2. Exploratory Data Analysis (EDA)
- **Correlation analysis**: Pearson correlation for numerical features
- **Bivariate analysis**:
  - Numeric vs Numeric (scatter plots)
  - Numeric vs Categorical (bar plots, box plots, distribution plots)
- **Eta correlation**: Measures association between categorical and numerical variables
- **Visualization**: Heatmaps, pair plots, scatter plots, and distribution comparisons

### 3. Feature Engineering
Creates derived features to capture complex patterns:
- `TotalPages`: Sum of all page types visited
- `TotalDuration`: Total time spent across all page types
- `ValuePerProduct`: Page value per product page visited
- `PageValuePerDuration`: Page value per unit time
- `ProductFocusRatio`: Ratio of product pages to total pages
- `ProductTimeRatio`: Interaction between product pages and duration
- `BounceExitOff`: Combined bounce and exit rate metric
- **Log transformations**: Applied to skewed features (skewness > 1)

### 4. Class Imbalance Handling
- **SMOTE (Synthetic Minority Over-sampling Technique)**:
  - Detects imbalance using a configurable threshold (default: 0.4)
  - Generates synthetic samples for minority class
  - Applied selectively (not to XGBoost, which uses `scale_pos_weight`)

### 5. Machine Learning Pipeline
- **Preprocessing Pipeline**:
  - Median imputation for missing values
  - StandardScaler for numerical features
  - Separate pipelines for different feature types
  - **Data leakage prevention**: Fit on training data only

### 6. Model Selection
Implements 7 classification algorithms:
1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **Random Forest Classifier**
4. **Gradient Boosting Classifier**
5. **AdaBoost Classifier**
6. **Bagging Classifier**
7. **XGBoost Classifier**

### 7. Hyperparameter Tuning
- **RandomizedSearchCV**: Efficient hyperparameter optimization
- **Model-specific parameter grids**: Tailored search spaces for each algorithm
- **Cross-validation**: 5-fold CV for robust evaluation
- **Scoring metric**: F1-score (balanced measure for imbalanced data)

### 8. Model Evaluation
Comprehensive metrics for each model:
- Training and test accuracy
- Precision, Recall, F1-score
- Confusion matrix
- Precision-Recall curve
- Cross-validation scores (mean and std)

## 📁 Project Structure

```
online_shoppers_intention/
│
├── data/
│   └── online_shoppers_intention.csv    # Dataset
│
├── src/
│   └── component/
│       └── classification.py            # Main classification class
│
├── requirements.txt                     # Python dependencies
├── .gitignore                          # Git ignore rules
└── README.md                           # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd online_shoppers_intention
```

2. **Create a virtual environment (recommended)**
```bash
python -m venv venv
```

3. **Activate the virtual environment**
   - Windows:
     ```bash
     venv\Scripts\activate
     ```
   - Linux/Mac:
     ```bash
     source venv/bin/activate
     ```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

## 📦 Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: Machine learning algorithms and tools
- **xgboost**: Gradient boosting framework
- **imbalanced-learn**: Tools for handling imbalanced datasets (SMOTE)
- **prefect**: Workflow orchestration (optional)

## 💻 Usage

### Basic Usage

```python
from src.component.classification import MyClassificationModel

# Initialize the model with data path
file_path = "data/online_shoppers_intention.csv"
mcm = MyClassificationModel(file_path=file_path)

# Load and preprocess data
df = mcm.load_data()
_, _, _, _, df_clean = mcm.missing_duplicate_value(df=df)

# Feature engineering
df_with_features = mcm.feature_enginnering(df=df_clean)

# Train-test split with preprocessing (no leakage)
X_train, X_test, y_train, y_test, X, y, preprocessor = mcm.train_test_split(df=df_with_features)

# Handle class imbalance
X_train_resampled, y_train_resampled = mcm.check_imbalanced(X_train, y_train)

# Get models
models = mcm.models(y_train=y_train)

# Hyperparameter tuning
tuned_results = mcm.tune_all_models_with_randomsearch(
    X_train=X_train_resampled, 
    y_train=y_train_resampled, 
    models=models, 
    n_iter=20, 
    cv=5, 
    scoring='f1'
)

# Evaluate models
best_models = {name: info["best_estimator"] for name, info in tuned_results.items()}
results = []
for name, model in best_models.items():
    result = mcm.evaluate_model(X_train_resampled, X_test, y_train_resampled, y_test, name, model)
    results.append(result)

# Print results
print(mcm.info_of_model(results))
```

### Running the Complete Pipeline

```bash
python src/component/classification.py
```

This executes the `main()` function which runs the entire pipeline end-to-end.

## 📈 Model Performance

The implementation evaluates models using multiple metrics:
- **Accuracy**: Overall correctness
- **Precision**: Positive prediction reliability
- **Recall**: Ability to find all positive cases
- **F1-Score**: Harmonic mean of precision and recall
- **Cross-validation**: 5-fold CV for generalization assessment

Models are tuned using RandomizedSearchCV to find optimal hyperparameters, balancing performance and computational efficiency.

## 🔍 Key Implementation Details

### Data Leakage Prevention
The implementation carefully prevents data leakage by:
1. Splitting data BEFORE any statistical transformations
2. Fitting preprocessors (scalers, encoders) ONLY on training data
3. Transforming test data using statistics learned from training data
4. Applying SMOTE only on training data

### XGBoost Special Handling
XGBoost uses `scale_pos_weight` parameter for class imbalance instead of SMOTE:
- Trained on original imbalanced data
- Weight automatically adjusts for class distribution
- More efficient than oversampling for tree-based models

### Correlation Analysis
- **Pearson correlation** for numeric-numeric relationships
- **Eta correlation** for categorical-numeric relationships
- Identifies top correlated feature pairs for deeper analysis

## 🎨 Visualization Capabilities

The class provides multiple visualization methods:
- Correlation heatmaps
- Pair plots for key features
- Scatter plots for bivariate analysis
- Bar plots and box plots for categorical comparisons
- Distribution plots comparing Revenue=True vs Revenue=False

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is open source and available under the MIT License.

## 👥 Authors

Project developed for online shopping behavior analysis and purchase prediction.

## 🙏 Acknowledgments

- Dataset source: Online Shoppers Purchasing Intention Dataset
- Libraries: scikit-learn, XGBoost, imbalanced-learn communities

## 📧 Contact

For questions or feedback, please open an issue in the repository.

---

**Note:** This is an educational/research project for demonstrating machine learning classification techniques on e-commerce data.
