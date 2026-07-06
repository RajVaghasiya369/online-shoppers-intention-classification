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


class MyClassificationModel:
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.list_of_model = []
        
    def load_data(self):
        """LOAD THE DATA"""
        
        try:
            df = pd.read_csv(f"{self.file_path}")
            
            return df
        
        except FileNotFoundError as e:
            return f"{e}"
    
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
            df_remove_duplicate = df.drop_duplicates(keep=False)
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
            raise Exception(f"{e}")
        
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
            raise Exception(f"{e}")
        
    def plot_corr(self, corr_matrix):
        try:
            plt.figure(figsize=(10, 12))
            sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
            plt.title("Correlation Matrix")
            plt.show()
            
        except Exception as e:
            raise Exception("No correlation matrix found.")
    
    def data_iqr(self,df):
        """REMOVE OUTLIERS FROM DATA"""
        
        try:
            num_cols = df.select_dtypes(include=['number']).columns.to_list()
            
            for col in num_cols:
                q1 = df[col].quantile(0.25)
                q3 = df[col].quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                median = df[col].median()
                
                outlier_count = ((df[col] < lower) | (df[col] > upper)).sum()
                
                if outlier_count > 0:
                    df[col] = df[col].apply(
                        lambda x: median if x < lower or x > upper else x
                    )
                    # print(f"'{col}': {outlier_count} outliers replaced with median ({median:.2f})")
                
            return df
            
        except Exception as e:
            raise Exception(f"IQR outlier removal failed: {e}")
        
    def bivariate_analysis_numeric_vs_numeric(self, df):
        """BIVARIATE ANALYSIS FOR NUMERIC VS NUMERIC"""
        
        """
        THIS THE MOST CORRELATED COLS
        BounceRates     ExitRates
        ProductRelated  ProductRelated_Duration
        Informational   Informational_Duration 
        Administrative  Administrative_Duration
                        ProductRelated 
        """
        
        try:
            plt.figure(figsize=(8,6))
            sns.scatterplot(x=df['BounceRates'], y=df['ExitRates'], data=df)
            plt.xlabel('BounceRates')
            plt.ylabel('ExitRates')
            plt.title("BounceRates VS ExitRates")
            
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
            raise Exception(f"{e}")
        
    def bivariate_analysis_numeric_vs_categories(self, df):
        """BIVARIATE ANALYSIS FOR NUMERIC VS CATEGORY"""
        
        """
        THIS THE MOST CORRELATED COLS
        Month   TrafficType
        VisitorType  ExitRates
        VisitorType  BounceRates
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
            sns.barplot(x=df['VisitorType'], y=df['BounceRates'], data=df)
            plt.xlabel('VisitorType')
            plt.ylabel('BounceRates')
            plt.title("VisitorType VS BounceRates")
                    
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
            sns.boxplot(x=df['VisitorType'], y=df['BounceRates'], data=df)
            plt.xlabel('VisitorType')
            plt.ylabel('BounceRates')
            plt.title("VisitorType VS BounceRates")
            
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
            sns.displot(df[df['Revenue'] == False]['BounceRates'])
            sns.displot(df[df['Revenue'] == True]['BounceRates'])
                        
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            raise Exception(f"{e}")    
    
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
            raise Exception(f"{e}")
    
    def train_test_split(self, df):
        """TRAIN TEST SPLIT THE DATA"""
        
        try:
            X = df.drop(columns=['Informational', 'SpecialDay', 'VisitorType_Other', 'Weekend', 'Month', 'Revenue'])
            y = df[['Revenue']]
            
            si = SimpleImputer(strategy='median')
            X_impute = si.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(X_impute, y, test_size=0.2, random_state=42)
            
            return X_train, X_test, y_train, y_test, X, y
    
        except Exception as e:
            raise Exception(f"{e}")
        
    def model_eval(self, X, y, name, X_train, X_test, y_train, y_test, model):
        """EVALUTING ALL MODEL FOR CLASSIFICATION"""
        
        try:
            if isinstance(name, str):
                pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scaler', StandardScaler()),
                    (name,model),
                ])
                
                pipeline.fit(X_train, y_train)
                y_pred = pipeline.predict(X_test)
                
                train_score = pipeline.score(X_train, y_train)
                test_score = pipeline.score(X_test, y_test)
                
                a = accuracy_score(y_test, y_pred)
                b = confusion_matrix(y_test, y_pred)
                c = precision_score(y_test, y_pred)
                d = recall_score(y_test, y_pred)
                e = f1_score(y_test, y_pred)
                f = precision_recall_curve(y_test, y_pred)
                
                cv = cross_val_score(pipeline, X, y, cv=5, scoring='accuracy')
                
                # importances = pd.Series(model.feature_importances_, index=X.columns)
                # , importances.sort_values(ascending=False).head(15)
                
                return train_score, test_score, a, b, c, d, e, f, cv.mean(), cv.std()
                
            else:
                return "Name must be string."
    
        except Exception as e:
            raise Exception(f"{e}")
    
    
def main():
    file_path = "../../data/online_shoppers_intention.csv"
    mcm = MyClassificationModel(file_path=file_path)
    df = mcm.load_data()
    info, stat = mcm.data_info(df=df)
    b_duplicate, a_duplicate, b_mis_val, af_mis_val, df_clean1 = mcm.missing_duplicate_value(df=df)
    corr_matrix, df_f= mcm.data_corr(df=df_clean1)
    
    df_clean = mcm.data_iqr(df=df_clean1)
    # print(type(df_clean))
    # print(df_clean['Revenue'].value_counts())
    
    # df_clean_fe = mcm.feature_eng(df=df_clean)
    
    df_clean_encoded = mcm.data_encoded(df=df_clean)
    # print(df_clean_encoded[['Month','Month_Encoded']])
    # print(df_clean_encoded[['Weekend','Revenue']])
    # print(df_clean_encoded.dtypes)
    
    X_train, X_test, y_train, y_test, X, y = mcm.train_test_split(df=df_clean_encoded)
    
    lr = LogisticRegression(C=0.03,max_iter=100,class_weight='balanced',random_state=42,solver='liblinear')
    rfc = RandomForestClassifier(n_estimators=300, criterion='entropy', max_depth=10, min_samples_split=5, min_samples_leaf=10, class_weight='balanced', random_state=42, n_jobs=-1)
    dtc = DecisionTreeClassifier(criterion='entropy', splitter='random',max_depth=20, random_state=42, class_weight='balanced')
    gbc = GradientBoostingClassifier(
        loss='log_loss',
        learning_rate=0.05,       # smaller learning rate
        n_estimators=200,         # more trees
        max_depth=4,              # deeper base learners
        min_samples_split=5,
        min_samples_leaf=2,
        subsample=0.8,            # stochastic boosting
        random_state=42
    )
    abc = AdaBoostClassifier(
        estimator=dtc,
        n_estimators=100,
        learning_rate=0.5,
        random_state=42
    )
    bc = BaggingClassifier(
        estimator=dtc,
        n_estimators=50,
        max_samples=0.8,        # 80% of samples per estimator
        max_features=0.8,       # 80% of features per estimator
        bootstrap=True,
        oob_score=True,         # enable out-of-bag evaluation
        random_state=42,
        n_jobs=-1
    )
    xgc = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        scale_pos_weight=10221/1908,
        eval_matric='logloss',
        max_depth=5,
        min_child_weight=3,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.01,
        reg_lambda=1.0,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1
    )
    
    train_score, test_score, acc, con, pre, rec, f1, pre_rec, mean, std= mcm.model_eval(X=X, y=y, name='LR',X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test, model=xgc)
    
    print(f"train score {train_score}")
    print(f"test score {test_score}")
    print(f"accuracy {acc}")
    print(f"confusion {con}")
    print(f"predision {pre}")
    print(f"recall {rec}")
    print(f"f1 score {f1}")
    print(f"predicion recall {pre_rec}")
    print(f"mean {mean}")
    print(f"std {std}")
    # print(f"fe\n {m}")
    # mcm.bivariate_analysis_numeric_vs_numeric(df_clean)
    # print(mcm.num_cat_cols(df=df_clean))
    # mcm.bivariate_analysis_numeric_vs_categories(df=df_clean)


    # print(df_clean.shape,end='\n')
    # mcm.plot_corr(corr_matrix=corr_matrix)
    # print(f"correletaion {corr_matrix}")
    # print(f"final pairs\n {df_f}")
    # print(f"before duplicate {b_duplicate}",end='\n')
    # print(f"after duplicate {a_duplicate}",end='\n')
    # print(f"before missing value {b_mis_val}",end='\n')
    # print(f"after missing value {af_mis_val}",end='\n')
    # print(f"size of data {df_clean.shape}",end='\n')
    
    
    
if __name__ == "__main__":
    main()