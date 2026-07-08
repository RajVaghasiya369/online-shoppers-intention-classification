import os
import pickle
import joblib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_val_score, RandomizedSearchCV) 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier, AdaBoostClassifier)
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve)
from sklearn.preprocessing import (StandardScaler, OneHotEncoder, OrdinalEncoder)
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from collections import Counter


class ModelSaver:
    """
    Handles saving and loading of trained models, preprocessors, and
    related metadata. Lives directly in this file so classification.py
    has no external dependency on a separate save_model.py module.
    """

    def __init__(self, base_dir=None):
        """
        Args:
            base_dir (str): Base directory for saving models. Defaults to
                a 'models' folder next to this file, so behavior no longer
                depends on the current working directory the script is run from.
        """
        if base_dir is None:
            base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        self.base_dir = base_dir
        self._ensure_directory_exists(self.base_dir)

    def _ensure_directory_exists(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)

    def save_model(self, model, model_name, method='joblib', subdir=None):
        """Save a trained model to disk. Returns the full filepath."""
        try:
            save_dir = os.path.join(self.base_dir, subdir) if subdir else self.base_dir
            self._ensure_directory_exists(save_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            if method == 'joblib':
                filename = f"{model_name}_{timestamp}.joblib"
                filepath = os.path.join(save_dir, filename)
                joblib.dump(model, filepath)
                print(f"Model saved using joblib: {filepath}")

            elif method == 'pickle':
                filename = f"{model_name}_{timestamp}.pkl"
                filepath = os.path.join(save_dir, filename)
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
                print(f"Model saved using pickle: {filepath}")

            else:
                raise ValueError(f"Unknown method: {method}. Use 'joblib' or 'pickle'")

            return filepath

        except Exception as e:
            raise Exception(f"Failed to save model '{model_name}': {e}") from e

    def load_model(self, filepath, method='joblib'):
        """Load a saved model from disk."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        try:
            if method == 'joblib':
                model = joblib.load(filepath)
                print(f"Model loaded using joblib: {filepath}")

            elif method == 'pickle':
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
                print(f"Model loaded using pickle: {filepath}")

            else:
                raise ValueError(f"Unknown method: {method}. Use 'joblib' or 'pickle'")

            return model

        except Exception as e:
            raise Exception(f"Failed to load model from '{filepath}': {e}") from e

    def save_preprocessor(self, preprocessor, name="preprocessor", method='joblib'):
        """Save a fitted preprocessor (e.g., ColumnTransformer, Pipeline)."""
        return self.save_model(preprocessor, name, method=method, subdir="preprocessors")

    def load_preprocessor(self, filepath, method='joblib'):
        """Load a saved preprocessor."""
        return self.load_model(filepath, method=method)

    def save_multiple_models(self, models_dict, method='joblib', subdir=None):
        """Save multiple models at once. Returns {model_name: filepath}."""
        saved_paths = {}
        for model_name, model in models_dict.items():
            saved_paths[model_name] = self.save_model(model, model_name, method=method, subdir=subdir)
        print(f"\nSaved {len(models_dict)} models successfully!")
        return saved_paths

    def save_model_metadata(self, model_name, metadata_dict, subdir=None):
        """Save model metadata (hyperparameters, metrics, training info) as JSON."""
        try:
            save_dir = os.path.join(self.base_dir, subdir) if subdir else self.base_dir
            self._ensure_directory_exists(save_dir)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{model_name}_metadata_{timestamp}.json"
            filepath = os.path.join(save_dir, filename)

            serializable_metadata = self._make_serializable(metadata_dict)

            with open(filepath, 'w') as f:
                json.dump(serializable_metadata, f, indent=4)

            print(f"Metadata saved: {filepath}")
            return filepath

        except Exception as e:
            raise Exception(f"Failed to save metadata for '{model_name}': {e}") from e

    def _make_serializable(self, obj):
        """Convert non-JSON-serializable objects to serializable format."""
        if isinstance(obj, dict):
            return {key: self._make_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (pd.Series, pd.DataFrame)):
            return obj.to_dict()
        elif hasattr(obj, 'tolist'):  # numpy arrays / numpy scalars
            return obj.tolist()
        elif isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        else:
            return str(obj)

    def load_model_metadata(self, filepath):
        """Load model metadata from a JSON file."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Metadata file not found: {filepath}")

        try:
            with open(filepath, 'r') as f:
                metadata = json.load(f)
            print(f"Metadata loaded: {filepath}")
            return metadata

        except Exception as e:
            raise Exception(f"Failed to load metadata from '{filepath}': {e}") from e

    def save_best_model(self, tuned_results, X_train, X_test, y_train, y_test,
                         metric='best_score', method='joblib'):
        """
        Save the best performing model based on a specified metric.
        Skips any entries whose metric is None (e.g. models that were
        never tuned), so it won't crash comparing None to a float.

        Returns: (best_model_name, saved_filepath, metadata_filepath)
        """
        try:
            scored_results = {
                name: info for name, info in tuned_results.items()
                if info.get(metric) is not None
            }
            if not scored_results:
                raise ValueError(f"No models have a non-None '{metric}' to compare.")

            best_model_name = max(scored_results, key=lambda k: scored_results[k][metric])
            best_model_info = scored_results[best_model_name]
            best_model = best_model_info['best_estimator']

            print(f"\nBest model: {best_model_name}")
            print(f"Best {metric}: {best_model_info[metric]:.4f}")

            model_filepath = self.save_model(
                best_model,
                f"best_model_{best_model_name}",
                method=method,
                subdir="best_models"
            )

            y_pred = best_model.predict(X_test)

            metadata = {
                'model_name': best_model_name,
                'best_params': best_model_info['best_params'],
                'cv_score': best_model_info['best_score'],
                'train_accuracy': best_model.score(X_train, y_train),
                'test_accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'selection_metric': metric,
            }

            metadata_filepath = self.save_model_metadata(
                f"best_model_{best_model_name}",
                metadata,
                subdir="best_models"
            )

            return best_model_name, model_filepath, metadata_filepath

        except Exception as e:
            raise Exception(f"Failed to save best model: {e}") from e

    def save_all_tuned_models(self, tuned_results, method='joblib'):
        """Save all tuned models from RandomizedSearchCV results, plus their metadata."""
        models_to_save = {
            name: info['best_estimator']
            for name, info in tuned_results.items()
        }

        saved_paths = self.save_multiple_models(
            models_to_save,
            method=method,
            subdir="tuned_models"
        )

        for name, info in tuned_results.items():
            metadata = {
                'model_name': name,
                'best_params': info['best_params'],
                'best_score': info['best_score'],
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
            self.save_model_metadata(name, metadata, subdir="tuned_models")

        return saved_paths

    def list_saved_models(self, subdir=None):
        """List all saved model files in the specified directory."""
        search_dir = os.path.join(self.base_dir, subdir) if subdir else self.base_dir

        if not os.path.exists(search_dir):
            print(f"Directory not found: {search_dir}")
            return []

        model_files = [
            os.path.join(search_dir, f) for f in os.listdir(search_dir)
            if f.endswith(('.joblib', '.pkl'))
        ]

        print(f"Found {len(model_files)} model(s) in {search_dir}")
        for file in model_files:
            print(f"  - {os.path.basename(file)}")

        return model_files

    def get_latest_model(self, model_name_prefix, subdir=None):
        """Get the most recently saved model matching a name prefix."""
        search_dir = os.path.join(self.base_dir, subdir) if subdir else self.base_dir

        if not os.path.exists(search_dir):
            raise FileNotFoundError(f"Directory not found: {search_dir}")

        matching_files = [
            os.path.join(search_dir, f)
            for f in os.listdir(search_dir)
            if f.startswith(model_name_prefix) and f.endswith(('.joblib', '.pkl'))
        ]

        if not matching_files:
            raise FileNotFoundError(f"No models found with prefix '{model_name_prefix}'")

        latest_file = max(matching_files, key=os.path.getmtime)
        print(f"Latest model: {latest_file}")

        return latest_file


class MyClassificationModel:
    
    def __init__(self, file_path, save_models=True, model_dir=None):
        """
        Args:
            file_path (str): Path to the input CSV.
            save_models (bool): Whether model-saving is enabled.
            model_dir (str): Where to save models. Defaults to a 'models'
                folder next to this file (see ModelSaver).
        """
        self.file_path = file_path
        self.list_of_model = []
        self.save_models = save_models
        
        # Initialize ModelSaver if saving is enabled
        if self.save_models:
            self.model_saver = ModelSaver(base_dir=model_dir)
            print(f"ModelSaver initialized. Models will be saved to: {self.model_saver.base_dir}")
        
    def load_data(self):
        """LOAD THE DATA"""
        try:
            df = pd.read_csv(f"{self.file_path}")
            
            return df
        
        except FileNotFoundError as e:
            raise f"{e}"
    
    def data_info(self, df):
        """INFORMATION ABOUT DATA"""
        try:
            shape = df.shape
            info = df.info()
            stat = df.describe()
            
            return (shape, stat)
        
        except Exception as e:
            raise Exception("No data found for information about data.") 
    
    def missing_duplicate_value(self,df):
        """HANDLE MISSING AND DUPLICATE VALUES"""
        try:
            before_duplicate = df.duplicated().sum()
            df_remove_duplicate = df.drop_duplicates(keep='first')
            after_duplicate = df_remove_duplicate.duplicated().sum()
            before_missing_balue = df_remove_duplicate.isnull().sum()
            df_clean = df_remove_duplicate.dropna()
            after_missing_value = df_clean.isnull().sum()
            
            return (before_duplicate, after_duplicate, before_missing_balue, after_missing_value, df_clean)
        
        except Exception as e:
            raise Exception("No data found for handle missing and duplicate values.")
    
    def data_corr(self, df):
        """CORRELATION ABOUT DATA"""
        try:
            num_col = df.select_dtypes(include=['number'])
            # print(num_col)
            corr_matrix = num_col.corr(method='pearson')
            
            pairs = corr_matrix.unstack()
            pairs = pairs[pairs.index.get_level_values(0) != pairs.index.get_level_values(1)]
            pairs.index = pairs.index.map(lambda x: tuple(sorted(x)))
            pairs = pairs[~pairs.index.duplicated()]

            pairs = pairs[abs(pairs) >= 0.4].sort_values(ascending=False, key=abs)
            final_pairs = pairs.head(20)
                        
            return corr_matrix, final_pairs
        
        except Exception:
            raise Exception("No data found for correlataion")
        
    def correlation_ratio(self, categories, values):
        """GETTING COLUMNS FOR NUMERICAL AND CATEGORICAL FOR PLOTTING"""
        try:
            categories = pd.Series(categories)
            values = pd.Series(values)
            
            df_temp = pd.DataFrame({'cat':categories, 'val':values}).dropna()
            
            cat_means = df_temp.groupby('cat')['val'].mean()
            overall_mean = df_temp['val'].mean()
            
            ss_between = sum(df_temp.groupby('cat').size() * (cat_means - overall_mean) ** 2)
            ss_total = sum((df_temp['val'] - overall_mean) ** 2)
            
            return np.sqrt(ss_between / ss_total) if ss_total != 0 else 0
        
        except Exception as e:
            raise Exception(f"{e}") from e
        
    def num_cat_cols(self, df):
        """ GETTING A BEST CORRELATION NUM AND CAT COLS """
        try:
            cat_cols = df.select_dtypes(include=['object']).columns.to_list()
            num_cols = df.select_dtypes(include=['number']).columns.to_list()

            results = []
            for cat in cat_cols:
                for num in num_cols:
                    eta = self.correlation_ratio(df[cat], df[num])
                    results.append((cat, num, eta))

            eta_df = pd.DataFrame(results, columns=['cat_col', 'num_col', 'eta'])
            eta_df = eta_df.sort_values('eta', ascending=False)

            return eta_df.head(10)
        
        except Exception as e:
            raise Exception(f"{e}") from e
        
    def plot_corr(self, corr_matrix, df):
        """PLOTTING CORRELATION"""
        try:
            plt.figure(figsize=(10, 12))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix")
            
            cols = ['PageValues', 'BounceRates', 'ExitRates', 'ProductRelated_Duration', 'Revenue']
            sns.pairplot(df[cols], hue='Revenue', corner=True)
            
            plt.show()
            
        except Exception as e:
            raise Exception("No correlation matrix found.")
    
    def data_iqr(self, df):
        """REMOVE OUTLIERS FROM DATA"""
        try:
            df = df.copy()  # avoid SettingWithCopyWarning / silent no-op on a view
            num_cols = df.select_dtypes(include=['number']).columns.to_list()

            for col in num_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                median = df[col].median()

                mask = (df[col] < lower) | (df[col] > upper)
                outlier_count = mask.sum()

                if outlier_count > 0:
                    df.loc[mask, col] = median
                    print(f"'{col}': capped {outlier_count} outliers "
                        f"(bounds: {lower:.2f} to {upper:.2f}, median: {median:.2f})")

            return df

        except Exception as e:
            raise Exception(f"IQR outlier removal failed: {e}") from e
        
    def bivariate_analysis_numeric_vs_numeric(self, df):
        """BIVARIATE ANALYSIS FOR NUMERIC VS NUMERIC"""
        
        """
        THIS THE MOST CORRELATED COLS
        BounceRatess     ExitRates
        ProductRelated  ProductRelated_Duration
        Informational   Informational_Duration 
        Administrative  Administrative_Duration
                        ProductRelated 
        """
        try:
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=df['BounceRatess'], y=df['ExitRates'], data=df)
            plt.xlabel('BounceRatess')
            plt.ylabel('ExitRates')
            plt.title("BounceRatess VS ExitRates")
            
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=df['ProductRelated'], y=df['ProductRelated_Duration'], data=df)
            plt.xlabel('ProductRelated')
            plt.ylabel('ProductRelated_Duration')
            plt.title("ProductRelated VS ProductRelated_Duration")
            
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=df['Informational'], y=df['Informational_Duration'], data=df)
            plt.xlabel('Informational')
            plt.ylabel('Informational_Duration')
            plt.title("Informational VS Informational_Duration")
            
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=df['Administrative'], y=df['Administrative_Duration'], data=df)
            plt.xlabel('Administrative')
            plt.ylabel('Administrative_Duration')
            plt.title("Administrative VS Administrative_Duration")
            
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=df['ProductRelated'], y=df['Administrative_Duration'], data=df)
            plt.xlabel('ProductRelated')
            plt.ylabel('Administrative_Duration')
            plt.title("ProductRelated VS Administrative_Duration")
            
            plt.tight_layout()
            plt.show()
                
        except Exception as e:
            raise Exception(f"{e}") from e
        
    def bivariate_analysis_numeric_vs_categories(self, df):
        """BIVARIATE ANALYSIS FOR NUMERIC VS CATEGORY"""
        
        """
        THIS THE MOST CORRELATED COLS
        Month   TrafficType
        VisitorType  ExitRates
        VisitorType  BounceRatess
        """
        try:

            plt.figure(figsize=(8,6))
            sns.barplot(x=df['Month'], y=df['TrafficType'], data=df)
            plt.xlabel('Month')
            plt.ylabel('TrafficType')
            plt.title("Month VS TrafficType")
        
            plt.figure(figsize=(8,6))
            sns.barplot(x=df['VisitorType'], y=df['ExitRates'], data=df)
            plt.xlabel('VisitorType')
            plt.ylabel('ExitRates')
            plt.title("VisitorType VS ExitRates")
                    
            plt.figure(figsize=(8,6))
            sns.barplot(x=df['VisitorType'], y=df['BounceRatess'], data=df)
            plt.xlabel('VisitorType')
            plt.ylabel('BounceRatess')
            plt.title("VisitorType VS BounceRatess")
                    
            plt.figure(figsize=(8,6))
            sns.boxplot(x=df['Month'], y=df['TrafficType'], data=df)
            plt.xlabel('Month')
            plt.ylabel('TrafficType')
            plt.title("Month VS TrafficType")
            
            plt.figure(figsize=(8,6))
            sns.boxplot(x=df['VisitorType'], y=df['ExitRates'], data=df)
            plt.xlabel('VisitorType')
            plt.ylabel('ExitRates')
            plt.title("VisitorType VS ExitRates")
            
            plt.figure(figsize=(8,6))
            sns.boxplot(x=df['VisitorType'], y=df['BounceRatess'], data=df)
            plt.xlabel('VisitorType')
            plt.ylabel('BounceRatess')
            plt.title("VisitorType VS BounceRatess")
            
            plt.figure(figsize=(8,6))
            sns.displot(df[df['Revenue'] == False]['Month'])
            sns.displot(df[df['Revenue'] == True]['Month'])
                        
            plt.figure(figsize=(8,6))
            sns.displot(df[df['Revenue'] == False]['TrafficType'])
            sns.displot(df[df['Revenue'] == True]['TrafficType'])
        
            plt.figure(figsize=(8,6))
            sns.displot(df[df['Revenue'] == False]['VisitorType'])
            sns.displot(df[df['Revenue'] == True]['VisitorType'])
            
            plt.figure(figsize=(8,6))
            sns.displot(df[df['Revenue'] == False]['ExitRates'])
            sns.displot(df[df['Revenue'] == True]['ExitRates'])
                        
            plt.figure(figsize=(8,6))
            sns.displot(df[df['Revenue'] == False]['BounceRatess'])
            sns.displot(df[df['Revenue'] == True]['BounceRatess'])
                    
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            raise Exception(f"{e}") from e  
        
    def feature_enginnering(self,df):
        """DOING FEATURE ENGINEERING FOR GETTING GOOD FEATURE"""
        try:
            df['TotalPages'] = (df['Administrative'] + df['Informational'] + df['ProductRelated']) 
            df['TotalDuration'] = (df['Administrative_Duration'] + df['Informational_Duration'] + df['ProductRelated_Duration'])
            df['ValuePerProduct'] = (df['PageValues'] / (df['ProductRelated'] + 1))
            df['PageValuePerDuration'] = (df['PageValues'] / (df['TotalPages'] + 1))
            df['ProductFocusRatio'] = (df['ProductRelated'] / (df['TotalPages'] + 1))
            df['ProductTimeRatio'] = (df['ProductRelated_Duration'] * df['ProductRelated'] + 1)
            df['BounceExitOff'] = (df['BounceRates'] + df['ExitRates'])
            
            skewed_cols = []
            for col in df.select_dtypes(include='number').columns:
                if col == "Revenue":
                    continue
                if df[col].skew() > 1:
                    df[col + '_log'] = np.log1p(df[col])
                    skewed_cols.append(col)

            return df
            
        except Exception as e:
            raise Exception(f"{e}") from e    
            
    def data_encoded(self,df):
        """CONVERT STRING OR OBJECT DATA TO NUMERIC BY ENCODING"""
        try:
            month_order = [['Feb','Mar','May','June','Jul','Aug','Sep','Oct','Nov','Dec']]
            oe = OrdinalEncoder(categories=month_order)
            df['Month_Encoded'] = oe.fit_transform(df[['Month']])
            
            ohe = OneHotEncoder(drop='first', sparse_output=False)
            encoded_array = ohe.fit_transform(df[['VisitorType']])
            encoded_df = pd.DataFrame(
                encoded_array,
                columns=ohe.get_feature_names_out(['VisitorType'])
            )
            
            df = pd.concat([df.drop(columns=['VisitorType']), encoded_df], axis=1)
        
            df['Weekend'] = np.where(df['Weekend'] == True, 1, 0)
            df['Revenue'] = np.where(df['Revenue'] == True, 1, 0)
        
            return df
        
        except Exception as e:
            raise Exception(f"{e}") from e
    
    def train_test_split(self, df):
        """SPLIT FIRST, THEN LEARN ALL STATISTICS FROM TRAIN ONLY"""
        try:
            # Deterministic, row-wise, NOT statistic-based — safe to do pre-split
            df['Weekend'] = np.where(df['Weekend'] == True, 1, 0)
            df['Revenue'] = np.where(df['Revenue'] == True, 1, 0)

            X = df.drop(columns=['Informational', 'SpecialDay',
                                'Weekend', 'Revenue'])
            y = df['Revenue']

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )

            num_cols = X.select_dtypes(include='number').columns.tolist()

            preprocessor = self.build_preprocessor(
                num_cols=num_cols, month_col='Month', visitor_col='VisitorType'
            )
            
            # Fit ONLY on train — this is the fix
            X_train_processed = preprocessor.fit_transform(X_train)
            # Transform only — test never influences learned statistics
            X_test_processed = preprocessor.transform(X_test)

            return (X_train_processed, X_test_processed, y_train, y_test,
                    X, y, preprocessor)

        except Exception as e:
            raise Exception(f"{e}") from e
        
    def build_preprocessor(self, num_cols, month_col='Month', visitor_col='VisitorType'):
        """
        Builds a single leakage-safe preprocessor.
        Fit this ONLY on X_train. Call .transform() (not .fit_transform()) on X_test.
        """
        month_order = [['Feb', 'Mar', 'May', 'June', 'Jul', 'Aug', 'Sep',
                            'Oct', 'Nov', 'Dec']]

        numeric_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ])
        
        month_pipeline = Pipeline(steps=[
                ('ordinal', OrdinalEncoder(
                    categories=month_order,
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
            )),
            ])

        visitor_pipeline = Pipeline(steps=[
                ('onehot', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore')),
            ])

        preprocessor = ColumnTransformer(transformers=[
                ('num', numeric_pipeline, num_cols),
                ('month', month_pipeline, [month_col]),
                ('visitor', visitor_pipeline, [visitor_col]),
            ], remainder='passthrough')

        return preprocessor    
                
    def check_imbalanced(self, X_train, y_train, threshold=0.4, sampling_strategy='auto'):
        """
        CHECK IF DATA IS IMBALANCED, APPLY SMOTE IF NEEDED
        
        threshold: minority/majority ratio below which data is considered imbalanced
                (0.4 means minority class is <40% of majority class)
        """
        try:
            class_counts = Counter(y_train)
            minority_class = min(class_counts, key=class_counts.get)
            majority_class = max(class_counts, key=class_counts.get)
            
            ratio = class_counts[minority_class] / class_counts[majority_class]
            
            print(f"Before SMOTE: {dict(class_counts)} | Imbalance ratio: {ratio:.3f}")
            
            if ratio < threshold:
                smote = SMOTE(random_state=42, sampling_strategy=sampling_strategy)
                X_train, y_train = smote.fit_resample(X_train, y_train)
                print(f"After SMOTE:  {dict(Counter(y_train))}")
            else:
                print("Data is reasonably balanced. Skipping SMOTE.")
            
            return X_train, y_train
        
        except Exception as e:
            raise Exception(f"{e}") from e
        
    def models(self, y_train):
        """ALL MODEL FOR CLASSIFICATION"""
        
        lr = LogisticRegression(C=0.3,max_iter=200,random_state=42,solver='liblinear')
        rfc = RandomForestClassifier(n_estimators=500, criterion='gini', max_depth=20, min_samples_split=2, min_samples_leaf=1, class_weight='balanced', random_state=42, n_jobs=-1)
        dtc = DecisionTreeClassifier(criterion='entropy', splitter='random',max_depth=20, random_state=42, class_weight='balanced', min_samples_leaf=1, min_samples_split=2)
        gbc = GradientBoostingClassifier(
                learning_rate=0.03,       
                n_estimators=150,         
                max_depth=5,              
                min_samples_split=5,
                min_samples_leaf=5,
                subsample=0.8,            
                random_state=42
            )
        abc = AdaBoostClassifier(
                estimator=dtc,
                n_estimators=200,
                learning_rate=0.5,
                random_state=42
            )
        bc = BaggingClassifier(
                estimator=dtc,
                n_estimators=75,
                max_features=1,       
                max_samples=0.8,        
                bootstrap=True,
                oob_score=True,         
                random_state=42,
            )
        xgbc = XGBClassifier(
                n_estimators=150,
                learning_rate=0.05,
                scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
                max_depth=7,
                min_child_weight=1,
                gamma=0.2,
                subsample=0.8,
                colsample_bytree=1.0,
                reg_alpha=0.1,
                reg_lambda=0.5,
                random_state=42,
                n_jobs=-1
            )
            
        models = {
                "LogisticRegression" : lr,
                "RandomForest" : rfc,
                "DecisionTree" : dtc,
                "GradientBoosting" : gbc,
                "AdaBoost" : abc,
                "BaggingClassifier" : bc,
                "XGB" : xgbc,
            }
            
        return models
        
    def get_param_distributions(self):
        """PARAMETER DISTRIBUTIONS FOR RANDOMIZEDSEARCHCV, KEYED BY MODEL NAME"""

        param_distributions = {
            "LogisticRegression": {
                "C": [0.001, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
                "solver": ["liblinear", "saga"],
                "max_iter": [100, 200, 500],
            },
            "RandomForest": {
                "n_estimators": [100, 200, 300, 400, 500],
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10, 20],
                "criterion": ["gini", "entropy"],
            },
            "DecisionTree": {
                "max_depth": [5, 10, 15, 20, None],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 5, 10],
                "criterion": ["gini", "entropy"],
                "splitter": ["best", "random"],
            },
            "GradientBoosting": {
                "n_estimators": [100, 150, 200, 300],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [2, 3, 4, 5],
                "min_samples_split": [2, 5, 10],
                "min_samples_leaf": [1, 2, 5],
                "subsample": [0.6, 0.8, 1.0],
            },
            "AdaBoost": {
                "n_estimators": [50, 100, 150, 200],
                "learning_rate": [0.01, 0.05, 0.1, 0.5, 1.0],
            },
            "BaggingClassifier": {
                "n_estimators": [10, 25, 50, 75, 100],
                "max_features": [0.5, 0.6, 0.8, 1.0],
                "max_samples": [0.5, 0.6, 0.8, 1.0],
            },
            "XGB": {
                "n_estimators": [100, 150, 200, 300],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [3, 4, 5, 6, 7],
                "min_child_weight": [1, 3, 5],
                "gamma": [0, 0.1, 0.2],
                "subsample": [0.6, 0.8, 1.0],
                "colsample_bytree": [0.6, 0.8, 1.0],
                "reg_alpha": [0, 0.01, 0.1],
                "reg_lambda": [0.5, 1.0, 1.5],
            },
        }

        return param_distributions

    def tune_model_with_randomsearch(self, X_train, y_train, name, model, param_distributions, n_iter=20, cv=5, scoring='f1', random_state=42, n_jobs=-1):
        """RUN RANDOMIZEDSEARCHCV FOR A SINGLE MODEL AND RETURN THE BEST ESTIMATOR"""

        try:
            search = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_distributions,
                n_iter=n_iter,
                scoring=scoring,
                cv=cv,
                random_state=random_state,
                n_jobs=n_jobs,
                verbose=0,
            )

            search.fit(X_train, y_train)

            print(f"[{name}] Best score ({scoring}): {search.best_score_:.4f}")
            print(f"[{name}] Best params: {search.best_params_}")

            return {
                "name": name,
                "best_estimator": search.best_estimator_,
                "best_params": search.best_params_,
                "best_score": search.best_score_,
            }

        except Exception as e:
            raise Exception(f"RandomizedSearchCV failed for {name}: {e}") from e

    def tune_all_models_with_randomsearch(self, X_train, y_train, models: dict, n_iter=20, cv=5, scoring='f1'):
        """LOOP OVER ALL MODELS, TUNE EACH WITH RANDOMIZEDSEARCHCV, RETURN BEST ESTIMATORS"""

        param_distributions = self.get_param_distributions()
        tuned_results = {}

        for name, model in models.items():
            params = param_distributions.get(name)

            if not params:
                print(f"[{name}] No param distribution defined, skipping tuning.")
                tuned_results[name] = {
                    "name": name,
                    "best_estimator": model,
                    "best_params": None,
                    "best_score": None,
                }
                continue

            result = self.tune_model_with_randomsearch(
                X_train=X_train,
                y_train=y_train,
                name=name,
                model=model,
                param_distributions=params,
                n_iter=n_iter,
                cv=cv,
                scoring=scoring,
            )
            tuned_results[name] = result

        return tuned_results

    def evaluate_model(self, X_train, X_test, y_train, y_test, name, model):
        """Evaluate a single model. X_train/X_test are ALREADY preprocessed."""
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            
            result = {
                name: {
                    'train_score': model.score(X_train, y_train),
                    'test_score': model.score(X_test, y_test),
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred),
                    'recall': recall_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'confusion_matrix': confusion_matrix(y_test, y_pred),
                    'precision_recall_curve': precision_recall_curve(y_test, y_pred),
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                }
            }
            return result

        except Exception as e:
            raise Exception(f"{e}") from e
        
    def run_all_models(self, X, y, X_train, X_test, y_train, y_test, models: dict):
        """RUN ALL MODEL"""
        
        for name, model in models.items():
            result = self.evaluate_model(X, y, name, X_train, X_test, y_train, y_test, model)
            self.list_of_model.append(result)
            
        return self.list_of_model
    
    def info_of_model(self, models: list):
        
        result = []
        
        for idx, model in enumerate(models):
            items = ", ".join([f"{k}: {v}" for k, v in model.items()])
            result.append(f"{idx} => {items}")
        
        return "\n".join(result)
    
    def save_trained_models(self, tuned_results, X_train, X_test, y_train, y_test, 
                           preprocessor=None, save_method='joblib'):
        """
        Save all trained models, preprocessor, and metadata.
        
        Args:
            tuned_results (dict): Dictionary of tuned model results
            X_train, X_test, y_train, y_test: Train and test datasets
            preprocessor: Fitted preprocessor object
            save_method (str): Serialization method ('joblib' or 'pickle')
            
        Returns:
            dict: Dictionary with saved file paths
        """
        if not self.save_models:
            print("Model saving is disabled. Set save_models=True to enable.")
            return None
        
        try:
            print("\n" + "="*60)
            print("SAVING MODELS AND METADATA")
            print("="*60)
            
            saved_info = {}
            
            # 1. Save all tuned models
            print("\n[1/3] Saving all tuned models...")
            saved_paths = self.model_saver.save_all_tuned_models(
                tuned_results, 
                method=save_method
            )
            saved_info['all_models'] = saved_paths
            
            # 2. Save the best model
            print("\n[2/3] Saving best model...")
            best_name, model_path, metadata_path = self.model_saver.save_best_model(
                tuned_results, 
                X_train, 
                X_test, 
                y_train, 
                y_test,
                metric='best_score',
                method=save_method
            )
            saved_info['best_model'] = {
                'name': best_name,
                'model_path': model_path,
                'metadata_path': metadata_path
            }
            
            # 3. Save preprocessor if provided
            if preprocessor is not None:
                print("\n[3/3] Saving preprocessor...")
                preprocessor_path = self.model_saver.save_preprocessor(
                    preprocessor, 
                    name="data_preprocessor",
                    method=save_method
                )
                saved_info['preprocessor'] = preprocessor_path
            else:
                print("\n[3/3] No preprocessor provided, skipping...")
            
            print("\n" + "="*60)
            print("ALL MODELS SAVED SUCCESSFULLY!")
            print("="*60)
            
            return saved_info
            
        except Exception as e:
            raise Exception(f"Failed to save models: {e}") from e
    
    def load_saved_model(self, model_path, method='joblib'):
        """
        Load a previously saved model.
        
        Args:
            model_path (str): Path to the saved model file
            method (str): Deserialization method
            
        Returns:
            Loaded model object
        """
        if not self.save_models:
            print("Warning: save_models is disabled, but you can still load models.")
        
        try:
            if not hasattr(self, 'model_saver'):
                self.model_saver = ModelSaver()
            
            return self.model_saver.load_model(model_path, method=method)
            
        except Exception as e:
            raise Exception(f"Failed to load model: {e}") from e
    
    def load_saved_preprocessor(self, preprocessor_path, method='joblib'):
        """
        Load a previously saved preprocessor.
        
        Args:
            preprocessor_path (str): Path to the saved preprocessor file
            method (str): Deserialization method
            
        Returns:
            Loaded preprocessor object
        """
        if not hasattr(self, 'model_saver'):
            self.model_saver = ModelSaver()
        
        try:
            return self.model_saver.load_preprocessor(preprocessor_path, method=method)
            
        except Exception as e:
            raise Exception(f"Failed to load preprocessor: {e}") from e

    
def main():
    """
    Main pipeline for training, evaluating, and saving models.
    """
    print("="*60)
    print("ONLINE SHOPPERS INTENTION PREDICTION PIPELINE")
    print("="*60)
    
    # Initialize with model saving enabled
    file_path = "../../data/online_shoppers_intention.csv"
    mcm = MyClassificationModel(file_path=file_path, save_models=True)

    # Load and preprocess data
    print("\n[STEP 1] Loading data...")
    df = mcm.load_data()
    print(f"Data loaded: {df.shape}")
    
    print("\n[STEP 2] Cleaning data (removing duplicates and missing values)...")
    _, _, _, _, df_clean1 = mcm.missing_duplicate_value(df=df)
    print(f"Clean data: {df_clean1.shape}")
    
    print("\n[STEP 3] Feature engineering...")
    df_with_feature = mcm.feature_enginnering(df=df_clean1)
    print(f"Data with engineered features: {df_with_feature.shape}")

    # Split + preprocess (fit only on train, transform test) — no leakage
    print("\n[STEP 4] Train-test split and preprocessing...")
    X_train, X_test, y_train, y_test, X, y, preprocessor = mcm.train_test_split(df=df_with_feature)
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")

    # Handle class imbalance with SMOTE
    print("\n[STEP 5] Handling class imbalance with SMOTE...")
    X_train_resampled, y_train_resampled = mcm.check_imbalanced(X_train=X_train, y_train=y_train)

    # Get models
    print("\n[STEP 6] Initializing models...")
    models = mcm.models(y_train=y_train)
    print(f"Models initialized: {list(models.keys())}")

    # --- RandomizedSearchCV hyperparameter tuning ---
    print("\n[STEP 7] Hyperparameter tuning with RandomizedSearchCV...")
    print("(This may take several minutes...)")
    
    # XGB tuned on the original imbalanced data (it uses scale_pos_weight),
    # everything else tuned on the SMOTE-resampled data.
    xgb_only = {"XGB": models["XGB"]}
    other_models = {name: model for name, model in models.items() if name != "XGB"}

    print("\n  - Tuning XGBoost (using scale_pos_weight)...")
    tuned_xgb = mcm.tune_all_models_with_randomsearch(
        X_train=X_train, y_train=y_train, models=xgb_only, n_iter=20, cv=5, scoring='f1'
    )
    
    print("\n  - Tuning other models (using SMOTE resampled data)...")
    tuned_others = mcm.tune_all_models_with_randomsearch(
        X_train=X_train_resampled, y_train=y_train_resampled, models=other_models, n_iter=20, cv=5, scoring='f1'
    )

    tuned_models = {**tuned_xgb, **tuned_others}
    best_models = {name: info["best_estimator"] for name, info in tuned_models.items()}
    # --- end tuning ---

    # Evaluate models
    print("\n[STEP 8] Evaluating models on test set...")
    list_of_model = []
    for name, model in best_models.items():
        if name == "XGB":
            # XGB uses scale_pos_weight — feed it the ORIGINAL imbalanced data
            result = mcm.evaluate_model(X_train, X_test, y_train, y_test, name, model)
        else:
            # Others use SMOTE-resampled data
            result = mcm.evaluate_model(X_train_resampled, X_test, y_train_resampled, y_test, name, model)
        list_of_model.append(result)

    # Print results
    print("\n" + "="*60)
    print("MODEL EVALUATION RESULTS")
    print("="*60)
    print(mcm.info_of_model(list_of_model))
    
    # Save all models, best model, and preprocessor
    print("\n[STEP 9] Saving models and preprocessor...")
    saved_info = mcm.save_trained_models(
        tuned_results=tuned_models,
        X_train=X_train_resampled,
        X_test=X_test,
        y_train=y_train_resampled,
        y_test=y_test,
        preprocessor=preprocessor,
        save_method='joblib'
    )
    
    # Print summary
    print("\n" + "="*60)
    print("PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*60)
    if saved_info:
        print(f"\nBest Model: {saved_info['best_model']['name']}")
        print(f"Best Model Path: {saved_info['best_model']['model_path']}")
        print(f"Metadata Path: {saved_info['best_model']['metadata_path']}")
        print(f"Preprocessor Path: {saved_info.get('preprocessor', 'N/A')}")
        print(f"\nTotal models saved: {len(saved_info['all_models'])}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()