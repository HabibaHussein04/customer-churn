import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
PLOT_DIR = BASE_DIR / "plots"
RESULT_DIR = BASE_DIR / "results"
MODEL_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)


def print_section(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)

print_section("Data Exploration")
df = pd.read_csv(DATA_DIR / "customer_churn.csv")
print(df.head())
print(df.info())
print("Number of duplicated rows:",df.duplicated().sum())
print(df.describe())

print_section("Data Cleaning")

df=df.drop(columns='customerID')
df['TotalCharges']=pd.to_numeric(df['TotalCharges'],errors='coerce')

for column in df.columns:
    print(f'\n{column}')
    print(df[column].unique())

#new column that represents how many additional services the customer uses
service_columns=['OnlineSecurity','OnlineBackup','DeviceProtection', 'TechSupport','StreamingTV','StreamingMovies']
df['TotalServices']=(df[service_columns]=='Yes').sum(axis=1)

#seeing if tech support affects churn depending on what type of internet the customer has
df['Internet_TechSupport']=(df['InternetService'].astype(str)+'_'+df['TechSupport'].astype(str))

#does it matter when a customer is subscribed from month to month and less that a year
df['New_MonthtoMonth']=((df['tenure']<=12) & (df['Contract']=='Month-to-month')).astype(int)

#does the number of additional services a senior citizen has affect the churn rate
df['Senior_TotalServices']=(df['SeniorCitizen']*df['TotalServices'])


df['CustomerTenureGroup'] = np.where(
    df['tenure'] <= 12,
    'New',
    'Established'
)

churn_rate = pd.crosstab(
    [df['CustomerTenureGroup'], df['Contract']],
    df['Churn'],
    normalize='index'
) * 100

print(churn_rate)

Y=df['Churn']
X=df.drop(columns='Churn')

plt.figure(figsize=(7, 5))
Y.value_counts().plot(kind='bar')
plt.xlabel('Churn')
plt.ylabel('Number of Customers')
plt.title('Churn Class Distribution')

plt.savefig(
    PLOT_DIR / "churn_class_distribution.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    Y,
    test_size=0.30,
    random_state=42,
    stratify=Y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=42,
    stratify=y_temp
)
y_train = y_train.map({'No': 0, 'Yes': 1})
y_val=y_val.map({'No': 0, 'Yes': 1})
y_test = y_test.map({'No': 0, 'Yes': 1})

numeric_original_columns = [
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges',

]
numeric_engineered_columns=[
    'New_MonthtoMonth',
    'Senior_TotalServices',
    'TotalServices',
    'SeniorCitizen',
    'tenure',
    'MonthlyCharges',
    'TotalCharges'
]
categorical_original_columns=[
    'gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod',

]
categorical_engineered_columns=[
    'Internet_TechSupport','gender',
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

def create_processor(numeric_columns,categorical_columns):
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat',
             OneHotEncoder(handle_unknown='ignore'),
             categorical_columns),

            ('num',
             Pipeline([
                 ('imputer', SimpleImputer(strategy='median')),
                 ('scaler', StandardScaler())
             ]),
             numeric_columns)
        ]
    )
    return preprocessor

preprocessor_original=create_processor(numeric_original_columns,categorical_original_columns)
preprocessor_engineered=create_processor(numeric_engineered_columns,categorical_engineered_columns)



print_section('Boxplot for outlier detection')
q1=X_train['tenure'].quantile(0.25)
q3=X_train['tenure'].quantile(0.75)
IQR=q3-q1
lower_bound=q1-1.5*IQR
upper_bound=q3+1.5*IQR
outliers=X_train[((X_train['tenure']<lower_bound) | ((X_train['tenure']>upper_bound)))]
print(outliers)

plt.figure(figsize=(7, 5))
plt.boxplot(X_train['tenure'])
plt.title('Tenure Boxplot for Outlier Detection')
plt.ylabel('Tenure')
plt.savefig(
    PLOT_DIR / "tenure_boxplot.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()

print("Q1:", q1)
print("Q3:", q3)
print("IQR:", IQR)
print("Lower bound:", lower_bound)
print("Upper bound:", upper_bound)
print("Number of outliers:", len(outliers))

contract_churn = pd.crosstab(
    df['Contract'],
    df['Churn'],
    normalize='index'
) * 100
ax = contract_churn.plot(
    kind='bar',
    stacked=True,
    figsize=(8, 5)
)

ax.set_title('Churn Rate by Contract Type')
ax.set_xlabel('Contract Type')
ax.set_ylabel('Percentage')
ax.legend(title='Churn')

plt.savefig(
    PLOT_DIR / "churn_rate_by_contract.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()


payment_churn = pd.crosstab(
    df['PaymentMethod'],
    df['Churn'],
    normalize='index'
) * 100
plt.figure(figsize=(10, 6))
payment_churn.plot(kind='bar')
plt.title('Churn Rate by Payment Method')
plt.xlabel('Payment Method')
plt.ylabel('Percentage')
plt.xticks(rotation=45)
plt.savefig(
    PLOT_DIR / "churn_rate_by_payment_method.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()


print_section("Models")

# =================================
# Logistic Regression baseline
# =================================

print_section("Logistic regression")
pipeline=Pipeline(steps=[('preprocessor',preprocessor_original),('classifier',LogisticRegression(max_iter=1000))])
pipeline.fit(X_train,y_train)
logistic_pred=pipeline.predict(X_test)
cm = confusion_matrix(y_test, logistic_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churn']
)
disp.plot()
plt.title('Logistic Regression Confusion Matrix')
plt.savefig(
    PLOT_DIR / "logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()
print('Precision:', precision_score(y_test, logistic_pred))
print('Recall:', recall_score(y_test, logistic_pred))
print('F1 Score:', f1_score(y_test, logistic_pred))
#save the scores
logistic_baseline_precision = precision_score(y_test, logistic_pred)
logistic_baseline_recall = recall_score(y_test, logistic_pred)
logistic_baseline_f1 = f1_score(y_test, logistic_pred)

# =================================
# Random Forrest baseline
# =================================

print('Random Forrest')
#random forrest
rf_pipeline=Pipeline(steps=[('preprocessor',preprocessor_original),('model',RandomForestClassifier(random_state=42))])
rf_pipeline.fit(X_train,y_train)
rf_baseline_pred=rf_pipeline.predict(X_test)
cm = confusion_matrix(y_test, rf_baseline_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churn']
)
disp.plot()
plt.title('Random Forest Confusion Matrix')
plt.savefig(
    PLOT_DIR / "random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()
print('Precision:', precision_score(y_test, rf_baseline_pred))
print('Recall:', recall_score(y_test, rf_baseline_pred))
print('F1 Score:', f1_score(y_test, rf_baseline_pred))
#save the scores
rf_baseline_precision = precision_score(y_test, rf_baseline_pred)
rf_baseline_recall = recall_score(y_test, rf_baseline_pred)
rf_baseline_f1 = f1_score(y_test, rf_baseline_pred)

# =================================
# XGBoost baseline
# =================================
print_section('XGBoost')
#scale_pos_weight=(y_train==0).sum()/(y_train==1).sum()
xg_pipeline=Pipeline(steps=[('preprocessor',preprocessor_original),('model',XGBClassifier(random_state=42))])
xg_pipeline.fit(X_train,y_train)
xg_baseline_pred=xg_pipeline.predict(X_test)
cm = confusion_matrix(y_test, xg_baseline_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churn']
)
disp.plot()
plt.title('XGBoost Confusion Matrix')
plt.savefig(
    PLOT_DIR / "xgboost_confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()
print('Precision:', precision_score(y_test, xg_baseline_pred))
print('Recall:', recall_score(y_test, xg_baseline_pred))
print('F1 Score:', f1_score(y_test, xg_baseline_pred))
xg_baseline_precision=precision_score(y_test,xg_baseline_pred)
xg_baseline_recall = recall_score(y_test, xg_baseline_pred)
xg_baseline_f1 = f1_score(y_test, xg_baseline_pred)

# =================================
# Catboost baseline
# =================================
print_section("CatBoost")
cat_model=CatBoostClassifier(random_state=42,cat_features=categorical_original_columns,verbose=0)
cat_pipeline=Pipeline(steps=[('model',cat_model)])
cat_X_train = X_train[numeric_original_columns + categorical_original_columns]
cat_X_test = X_test[numeric_original_columns + categorical_original_columns]
cat_pipeline.fit(cat_X_train,y_train)

cat_baseline_pred=cat_pipeline.predict(cat_X_test)

cat_baseline_precision=precision_score(y_test,cat_baseline_pred)
cat_baseline_recall = recall_score(y_test, cat_baseline_pred)
cat_baseline_f1 = f1_score(y_test, cat_baseline_pred)
cm = confusion_matrix(y_test, cat_baseline_pred)
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churn']
)
disp.plot()
plt.title('CatBoost Confusion Matrix')
plt.savefig(
    PLOT_DIR / "catboost_confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)
plt.close()
print("Precision:", cat_baseline_precision)
print("Recall:", cat_baseline_recall)
print("F1 Score:", cat_baseline_f1)

# =================================
# Logistic Regression Hyperparameter Tuning
# =================================

print_section("Logistic Regression Hyperparameter Tuning")
logistic_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_engineered),
    ('model', LogisticRegression(max_iter=1000))
])
logistic_params = {
    'model__C': [0.01, 0.1, 1, 10, 100],
    'model__l1_ratio': [0, 1],
    'model__class_weight': [None, 'balanced'],
    'model__solver': ['liblinear']
}
logistic_grid=GridSearchCV(logistic_pipeline,logistic_params,cv=5,scoring='f1',n_jobs=-1)
logistic_grid.fit(X_train,y_train)
print("Best Logistic parameters:", logistic_grid.best_params_)
print("Best Logistic CV F1:", logistic_grid.best_score_)
logistic_best = logistic_grid.best_estimator_
joblib.dump(logistic_best, MODEL_DIR / "logistic_best.pkl")
logistic_tuned_pred = logistic_best.predict(X_test)
print("Precision:", precision_score(y_test, logistic_tuned_pred))
print("Recall:", recall_score(y_test, logistic_tuned_pred))
print("F1:", f1_score(y_test, logistic_tuned_pred))
#saves the scores
logistic_tuned_precision = precision_score(y_test, logistic_tuned_pred)
logistic_tuned_recall = recall_score(y_test, logistic_tuned_pred)
logistic_tuned_f1 = f1_score(y_test, logistic_tuned_pred)

# =================================
# Random Forrest Hyperparameter Tuning
# =================================
print_section("Random Forrest Hyperparameter Tuning")
rf_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_engineered),
    ('model', RandomForestClassifier(random_state=42))
])
rf_params={
    'model__n_estimators':[100, 200, 300],
    'model__max_depth':[None,10,20,30],
    'model__min_samples_split':[2,5,10],
    'model__min_samples_leaf':[1,2,4]
}
rf_grid=GridSearchCV(rf_pipeline,rf_params,cv=5,scoring='f1',n_jobs=-1)
rf_grid.fit(X_train,y_train)
print("Best RF parameters:", rf_grid.best_params_)
print("Best RF CV F1:", rf_grid.best_score_)
rf_best = rf_grid.best_estimator_
joblib.dump(rf_best, MODEL_DIR / "random_forest_best.pkl")
rf_probabilities = rf_best.predict_proba(X_test)[:, 1]

rf_tuned_pred = rf_best.predict(X_test)

print("Precision:", precision_score(y_test, rf_tuned_pred))
print("Recall:", recall_score(y_test, rf_tuned_pred))
print("F1:", f1_score(y_test, rf_tuned_pred))
#saves the scores
rf_tuned_precision = precision_score(y_test, rf_tuned_pred)
rf_tuned_recall = recall_score(y_test, rf_tuned_pred)
rf_tuned_f1 = f1_score(y_test, rf_tuned_pred)

# =================================
# XGBoost Hyperparameter Tuning
# =================================

print_section("XGBoost Hyperparameter Tuning")
xg_pipeline=Pipeline(steps=[('preprocessor',preprocessor_engineered),('model',XGBClassifier(random_state=42))])
xg_params={
    'model__n_estimators': [100, 200],
    'model__max_depth': [3, 5],
    'model__learning_rate': [0.05, 0.1],
}
xg_grid=GridSearchCV(xg_pipeline,xg_params,cv=5,scoring='f1',n_jobs=-1)
xg_grid.fit(X_train,y_train)
print("Best XGBoost parameters:", xg_grid.best_params_)
print("Best XGBoost CV F1:", xg_grid.best_score_)
xg_best = xg_grid.best_estimator_
joblib.dump(xg_best, MODEL_DIR / "xgboost_best.pkl")
xg_tuned_pred = xg_best.predict(X_test)

print("Precision:", precision_score(y_test, xg_tuned_pred))
print("Recall:", recall_score(y_test, xg_tuned_pred))
print("F1:", f1_score(y_test, xg_tuned_pred))
xg_tuned_precision = precision_score(y_test, xg_tuned_pred)
xg_tuned_recall = recall_score(y_test, xg_tuned_pred)
xg_tuned_f1 = f1_score(y_test, xg_tuned_pred)

# =================================
# CatBoost Hyperparameter Tuning
# =================================
print_section("CatBoost Hyperparameter Tuning")
cat_model = CatBoostClassifier(
    random_state=42,
    verbose=0,
    thread_count=4
)

cat_params = {
    'iterations': [200, 400],
    'depth': [4, 6],
    'learning_rate': [0.05, 0.1],
    'l2_leaf_reg': [1, 3]
}

cat_grid = GridSearchCV(
    cat_model,
    cat_params,
    cv=3,
    scoring='f1',
    n_jobs=-1
)
cat_X_train = X_train[
    numeric_engineered_columns + categorical_engineered_columns
]

cat_grid.fit(
    cat_X_train,
    y_train,
    cat_features=categorical_engineered_columns
)
print("Best CatBoost parameters:", cat_grid.best_params_)
print("Best CatBoost CV F1:", cat_grid.best_score_)

cat_best = cat_grid.best_estimator_
joblib.dump(cat_best, MODEL_DIR / "catboost_best.pkl")

cat_X_test = X_test[
    numeric_engineered_columns + categorical_engineered_columns
]

cat_tuned_pred = cat_best.predict(cat_X_test)

print("Precision:", precision_score(y_test, cat_tuned_pred))
print("Recall:", recall_score(y_test, cat_tuned_pred))
print("F1:", f1_score(y_test, cat_tuned_pred))
cat_tuned_precision = precision_score(y_test, cat_tuned_pred)
cat_tuned_recall = recall_score(y_test, cat_tuned_pred)
cat_tuned_f1 = f1_score(y_test, cat_tuned_pred)


# ==============================
# SVM Hyperparameter Tuning
# ==============================
print_section("SVM Hyperparameter Tuning")

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_engineered),
    ('model', SVC(random_state=42))
])

svm_params = {
    'model__C': [0.1, 1, 10],
    'model__gamma': ['scale', 'auto'],
    'model__kernel': ['rbf', 'linear'],
    'model__class_weight': [None, 'balanced']
}

svm_grid = GridSearchCV(
    svm_pipeline,
    svm_params,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

svm_grid.fit(X_train, y_train)

print("Best SVM parameters:", svm_grid.best_params_)
print("Best SVM CV F1:", svm_grid.best_score_)

svm_best = svm_grid.best_estimator_
svm_tuned_pred = svm_best.predict(X_test)

svm_tuned_precision = precision_score(y_test, svm_tuned_pred)
svm_tuned_recall = recall_score(y_test, svm_tuned_pred)
svm_tuned_f1 = f1_score(y_test, svm_tuned_pred)

joblib.dump(
    svm_best,
    MODEL_DIR / "svm_best.pkl"
)

## ==============================
# Gradient Boosting Hyperparameter Tuning
# ==============================

print_section("Gradient Boosting Hyperparameter Tuning")

gb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor_engineered),
    ('model', GradientBoostingClassifier(random_state=42))
])

gb_params = {
    'model__n_estimators': [100, 200],
    'model__learning_rate': [0.05, 0.1],
    'model__max_depth': [2, 3],
    'model__min_samples_leaf': [1, 5, 10]
}

gb_grid = GridSearchCV(
    gb_pipeline,
    gb_params,
    cv=5,
    scoring='f1',
    n_jobs=-1
)

gb_grid.fit(X_train, y_train)

print(
    "Best Gradient Boosting parameters:",
    gb_grid.best_params_
)

print(
    "Best Gradient Boosting CV F1:",
    gb_grid.best_score_
)

gb_best = gb_grid.best_estimator_

joblib.dump(
    gb_best,
    MODEL_DIR / "gradient_boosting_best.pkl"
)

gb_pred = gb_best.predict(X_test)

gb_precision = precision_score(y_test, gb_pred)
gb_recall = recall_score(y_test, gb_pred)
gb_f1 = f1_score(y_test, gb_pred)

print("Precision:", gb_precision)
print("Recall:", gb_recall)
print("F1:", gb_f1)




print_section("Threshold Optimization")
# ==============================
# Logistic Regression Threshold Tuning
# ==============================

print_section("Logistic Regression Threshold Tuning")

# Get churn probabilities for the VALIDATION set
logistic_val_probabilities = logistic_best.predict_proba(X_val)[:, 1]

best_f1 = 0
logistic_best_threshold = 0.5

# Try different thresholds
for threshold in np.arange(0.10, 0.71, 0.01):

    predictions = (
        logistic_val_probabilities >= threshold
    ).astype(int)

    f1 = f1_score(y_val, predictions)

    if f1 > best_f1:
        best_f1 = f1
        logistic_best_threshold = threshold

print("Best Logistic Regression threshold:", logistic_best_threshold)
print("Validation F1:", best_f1)


# =================================
# Final evaluation on TEST set
# =================================

logistic_test_probabilities = logistic_best.predict_proba(X_test)[:, 1]

logistic_final_pred = (
    logistic_test_probabilities >= logistic_best_threshold
).astype(int)

logistic_final_precision = precision_score(
    y_test,
    logistic_final_pred
)

logistic_final_recall = recall_score(
    y_test,
    logistic_final_pred
)

logistic_final_f1 = f1_score(
    y_test,
    logistic_final_pred
)

print("Final Test Precision:", logistic_final_precision)
print("Final Test Recall:", logistic_final_recall)
print("Final Test F1:", logistic_final_f1)


# ==============================
# Random Forest Threshold Tuning
# ==============================

print_section("Random Forest Threshold Tuning")

rf_val_probabilities = rf_best.predict_proba(X_val)[:, 1]

best_f1 = 0
rf_best_threshold = 0.5

for threshold in np.arange(0.10, 0.71, 0.01):

    predictions = (
        rf_val_probabilities >= threshold
    ).astype(int)

    f1 = f1_score(y_val, predictions)

    if f1 > best_f1:
        best_f1 = f1
        rf_best_threshold = threshold

print("Best RF threshold:", rf_best_threshold)
print("Validation F1:", best_f1)


# Final test evaluation

rf_test_probabilities = rf_best.predict_proba(X_test)[:, 1]

rf_final_pred = (
    rf_test_probabilities >= rf_best_threshold
).astype(int)

rf_final_precision = precision_score(y_test, rf_final_pred)
rf_final_recall = recall_score(y_test, rf_final_pred)
rf_final_f1 = f1_score(y_test, rf_final_pred)

print("Final Test Precision:", rf_final_precision)
print("Final Test Recall:", rf_final_recall)
print("Final Test F1:", rf_final_f1)


# ==============================
# XGBoost Threshold Tuning
# ==============================

print_section("XGBoost Threshold Tuning")

xg_val_probabilities = xg_best.predict_proba(X_val)[:, 1]

best_f1 = 0
xg_best_threshold = 0.5

for threshold in np.arange(0.10, 0.71, 0.01):

    predictions = (
        xg_val_probabilities >= threshold
    ).astype(int)

    f1 = f1_score(y_val, predictions)

    if f1 > best_f1:
        best_f1 = f1
        xg_best_threshold = threshold

print("Best XGBoost threshold:", xg_best_threshold)
print("Validation F1:", best_f1)

xg_test_probabilities = xg_best.predict_proba(X_test)[:, 1]

xg_final_pred = (
    xg_test_probabilities >= xg_best_threshold
).astype(int)

xg_final_precision = precision_score(y_test, xg_final_pred)
xg_final_recall = recall_score(y_test, xg_final_pred)
xg_final_f1 = f1_score(y_test, xg_final_pred)

print("Final Test Precision:", xg_final_precision)
print("Final Test Recall:", xg_final_recall)
print("Final Test F1:", xg_final_f1)


# ==============================
# CatBoost Threshold Tuning
# ==============================

print_section("CatBoost Threshold Tuning")

cat_X_val = X_val[
    numeric_engineered_columns + categorical_engineered_columns
]

cat_val_probabilities = cat_best.predict_proba(cat_X_val)[:, 1]

best_f1 = 0
cat_best_threshold = 0.5

for threshold in np.arange(0.10, 0.71, 0.01):

    predictions = (
        cat_val_probabilities >= threshold
    ).astype(int)

    f1 = f1_score(y_val, predictions)

    if f1 > best_f1:
        best_f1 = f1
        cat_best_threshold = threshold

print("Best CatBoost threshold:", cat_best_threshold)
print("Validation F1:", best_f1)


cat_X_test = X_test[
    numeric_engineered_columns + categorical_engineered_columns
]

cat_test_probabilities = cat_best.predict_proba(cat_X_test)[:, 1]

cat_final_pred = (
    cat_test_probabilities >= cat_best_threshold
).astype(int)

cat_final_precision = precision_score(y_test, cat_final_pred)
cat_final_recall = recall_score(y_test, cat_final_pred)
cat_final_f1 = f1_score(y_test, cat_final_pred)

print("Final Test Precision:", cat_final_precision)
print("Final Test Recall:", cat_final_recall)
print("Final Test F1:", cat_final_f1)


thresholds = {
    "logistic_regression": logistic_best_threshold,
    "random_forest": rf_best_threshold,
    "xgboost": xg_best_threshold,
    "catboost": cat_best_threshold
}


# ==============================
# Gradient Boosting Threshold Optimization
# ==============================

# Get decision scores on validation set
gb_val_scores = gb_best.predict_proba(X_val)[:, 1]

best_f1 = 0
gb_best_threshold = 0.5

# Test thresholds from 0.10 to 0.90
for threshold in np.arange(0.10, 0.91, 0.01):

    gb_val_pred = (gb_val_scores >= threshold).astype(int)

    f1 = f1_score(y_val, gb_val_pred)

    if f1 > best_f1:
        best_f1 = f1
        gb_best_threshold = threshold

print("\nGradient Boosting Threshold Optimization")
print("Best threshold:", gb_best_threshold)
print("Validation F1:", best_f1)


gb_test_scores = gb_best.predict_proba(X_test)[:, 1]

gb_final_pred = (
    gb_test_scores >= gb_best_threshold
).astype(int)

gb_final_precision = precision_score(y_test, gb_final_pred)
gb_final_recall = recall_score(y_test, gb_final_pred)
gb_final_f1 = f1_score(y_test, gb_final_pred)

print("\nGradient Boosting Final Test Results")
print("Precision:", gb_final_precision)
print("Recall:", gb_final_recall)
print("F1:", gb_final_f1)




# ==============================
# SVM Threshold Optimization
# ==============================

print_section("SVM Threshold Optimization")

# Get decision scores from the VALIDATION set
svm_val_scores = svm_best.decision_function(X_val)

best_f1 = 0
svm_best_threshold = 0.0

# Try different decision-score thresholds
for threshold in np.arange(-2.0, 2.01, 0.01):

    predictions = (
        svm_val_scores >= threshold
    ).astype(int)

    f1 = f1_score(y_val, predictions)

    if f1 > best_f1:
        best_f1 = f1
        svm_best_threshold = threshold

print("Best SVM threshold:", svm_best_threshold)
print("Validation F1:", best_f1)


svm_test_scores = svm_best.decision_function(X_test)

svm_final_pred = (
    svm_test_scores >= svm_best_threshold
).astype(int)

svm_final_precision = precision_score(
    y_test,
    svm_final_pred
)

svm_final_recall = recall_score(
    y_test,
    svm_final_pred
)

svm_final_f1 = f1_score(
    y_test,
    svm_final_pred
)

print("Final Test Precision:", svm_final_precision)
print("Final Test Recall:", svm_final_recall)
print("Final Test F1:", svm_final_f1)


cm = confusion_matrix(y_test, svm_final_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churn']
)

disp.plot()

plt.title('SVM Confusion Matrix')

plt.savefig(
    PLOT_DIR / "svm_confusion_matrix.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()

# ==============================
# Final Model Comparison
# ==============================

print_section("Final Model Comparison")

results = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Random Forest',
        'XGBoost',
        'CatBoost',
        'SVM',
        'Gradient Boosting'
    ],

    'Baseline F1': [
        logistic_baseline_f1,
        rf_baseline_f1,
        xg_baseline_f1,
        cat_baseline_f1,
        np.nan,
        np.nan
    ],

    'Tuned F1': [
        logistic_tuned_f1,
        rf_tuned_f1,
        xg_tuned_f1,
        cat_tuned_f1,
        svm_tuned_f1,
        gb_f1
    ],

    'Final Precision': [
        logistic_final_precision,
        rf_final_precision,
        xg_final_precision,
        cat_final_precision,
        svm_final_precision,
        gb_final_precision
    ],

    'Final Recall': [
        logistic_final_recall,
        rf_final_recall,
        xg_final_recall,
        cat_final_recall,
        svm_final_recall,
        gb_final_recall
    ],

    'Final F1': [
        logistic_final_f1,
        rf_final_f1,
        xg_final_f1,
        cat_final_f1,
        svm_final_f1,
        gb_final_f1
    ],

    'Threshold': [
        logistic_best_threshold,
        rf_best_threshold,
        xg_best_threshold,
        cat_best_threshold,
        svm_best_threshold,
        gb_best_threshold
    ]
})
joblib.dump(thresholds, MODEL_DIR / "thresholds.pkl")


print(results)

results.to_csv(
    RESULT_DIR / "results.csv",
    index=False
)
plt.figure(figsize=(10, 6))

plt.bar(
    results['Model'],
    results['Final F1']
)

plt.title('Final Model F1 Comparison')
plt.xlabel('Model')
plt.ylabel('F1 Score')
plt.xticks(rotation=25)

plt.savefig(
    PLOT_DIR / "final_model_comparison.png",
    dpi=300,
    bbox_inches='tight'
)

plt.close()