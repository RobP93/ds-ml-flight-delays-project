# %%
# SESSION SETUP ================================================================
# Install packages
# !pip install xgboost
# !pip install catboost

# Import packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import datetime as dt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, mean_squared_error, roc_auc_score, mean_absolute_error, r2_score
from catboost import CatBoostClassifier

# Load data
flights_import = pd.read_csv('./data/flights_train_cleaned.csv')
flights_test = pd.read_csv('./data/Test.csv')    # Unprocessed!

# Random Seed
RSEED = 50

# %%
# DATA PRE-PROCESSING ==========================================================
# Copy data
flights = flights_import.copy()

# Fix format of date columns
flights['datop'] = pd.to_datetime(flights['datop'])
flights['std'] = pd.to_datetime(flights['std'])
flights['sta'] = pd.to_datetime(flights['sta'])

# Fix format of categorical columns
flights['depstn'] = flights['depstn'].astype('category')
flights['arrstn'] = flights['arrstn'].astype('category')
flights['status'] = flights['status'].astype('category')
flights['dep_country'] = flights['dep_country'].astype('category')
flights['arr_country'] = flights['arr_country'].astype('category')
flights['od'] = flights['od'].astype('category')
flights['airline'] = flights['airline'].astype('category')
flights['flt_type'] = flights['flt_type'].astype('category')
flights['daypart'] = flights['daypart'].astype('category')
flights['year'] = flights['datop'].dt.strftime('%Y')
flights['year'] = flights['year'].astype('int')

# -- DON'T DROP! Actually deteriorated performance 
# Dropping features
# flights = flights.drop(['datop', 'dep_country', 'arr_country', 'std', 'sta'],
#                        axis=1)

# SET CUT OFF FOR 500 MIN DELAY
flights = flights.loc[flights['target'] <= 500, :]

# Reorder columns
flights = flights.reindex(columns=[
    'target', 'depstn', 'arrstn', 'od', 'airline', 'status', 'flt_type',
    'scheduled_duration_min', 'year', 'month', 'weekofyear', 'is_weekend',
    'dep_hour', 'arr_hour', 'daypart'])

# Define features and target
features = flights.drop('target', axis=1)
target_num = flights['target']
target_cat = np.where(flights['target'] > 15, 1, 0)
target_cat = pd.Series(target_cat).astype('bool')        # 1 = Delayed

# Cast features into separate objects by dtype and extract column 
# names (needed as reference for xgb_pipe since it uses pd.df input)
cat_features = features.select_dtypes(include='category').columns
num_features = features.select_dtypes(include='number').columns

# not needed, already in num_features:
# dtm_features = features.select_dtypes(include='datetime').drop('datop', axis=1).columns        


# %%
# TRAIN-TEST-SPLIT FOR CLASSIFICATION ==========================================
X_train, X_test, y_train, y_test = train_test_split(
    features, target_cat, test_size=0.3, random_state=RSEED)

# Drop datetime columns since they're not compatible with XGBoost.
# Datetimes were converted to numeric components.
X_train = X_train.select_dtypes(exclude='datetime')
X_test = X_test.select_dtypes(exclude='datetime')

print('X_train shape:', X_train.shape)
print('X_test shape :', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape: ', y_test.shape)


# %%
# BUILD PIPELINES ==============================================================
# Pre-processing pipelines
cat_pipeline = Pipeline([
    ('1hot', OneHotEncoder(handle_unknown='ignore'))
])

num_pipeline = Pipeline([
    ('standard_scaler', StandardScaler())
])

preprocessor_pipe = ColumnTransformer([
    ('cat', cat_pipeline, cat_features),
    ('num', num_pipeline, num_features)
], remainder='passthrough')

# XGBoost pipeline
# XGBoost classifier
xgb_pipe = Pipeline([
    ('preprocessor', preprocessor_pipe),
    ('xgb_classification', xgb.XGBClassifier(
          n_estimators = 100,
          learning_rate = 0.1,
          max_depth = 2,
          objective = 'binary:logistic',
          random_state = RSEED
    ))
])

# Note on eval_metric parameter:
# "All scorer objects follow the convention that higher return values are better
# than lower return values. Thus metrics which measure the distance between the
# model and the data, like metrics.mean_squared_error, are available as
# ‘neg_mean_squared_error’ which return the negated value of the metric."
# https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-string-names

# %%
# TRAIN XGBOOST CLASSIFIER =====================================================
# Train model
xgb_pipe.fit(X_train, y_train)


# %%
# EVALUATE XGBOOST CLASSIFIER ==================================================
# Write function to easily compute key metrics for models
def evaluate_model(model, pipe, X_train, y_train, X_test, y_test):
    """
    Evalutates the performance of piped machine learning models.

    Args:
        model (str):
            *'xgb'* for XGBoost model\n
            *'ln'* for linear regression model\n
            *'cb'* for CatBoost classifier
        pipe (pipeline object): Model pipeline object
        X_train (dataframe): Matrix of values for training features
        y_train (array): Array of values for training label
        X_test (dataframe): Matrix of values for test features
        y_test (array): Array of values for test label
    
    Returns:
        Text and object output of RMSE, accuracy, recall and precision of the
        model. Additionally, the predicted y-values are outputted to the global
        environment.
    """

    if model == 'xgb':
        
        # Fit model and predict y-values
        pipe.fit(X_train, y_train)
        y_pred_xgb = xgb_pipe.predict(X_test)

        # Compute model metrics
        xgb_rmse = np.sqrt(mean_squared_error(y_test, y_pred_xgb))
        xgb_accuracy = accuracy_score(y_test, y_pred_xgb)
        xgb_recall = recall_score(y_test, y_pred_xgb)
        xgb_precision = precision_score(y_test, y_pred_xgb)
        xgb_auc = roc_auc_score(y_test, y_pred_xgb)

        # Print results
        print('Accuracy of XGBoost classifier:  ', round(xgb_accuracy, 2))
        print('Recall of XGBoost classifier:    ', round(xgb_recall, 2))
        print('Precision of XGBoost classifier: ', round(xgb_precision, 2))
        print('AUC score of XGBoost classifier: ', round(xgb_auc, 2))

        # Return evaluation metrics as variables
        return {
            'xgb_rmse': xgb_rmse,
            'xgb_accuracy': xgb_accuracy,
            'xgb_recall': xgb_recall,
            'xgb_precision': xgb_precision,
            'xgb_auc': xgb_auc
        }
    
    elif model == 'ln':

        # Fit model and predict y-values
        pipe.fit(X_train, y_train)
        y_pred_ln = ln_pipe.predict(X_test)

        # Compute model metrics
        ln_rmse = np.sqrt(mean_squared_error(y_test, y_pred_ln))
        ln_mae = mean_absolute_error(y_test, y_pred_ln)
        ln_r2 = r2_score(y_test, y_pred_ln)

        # Print results
        print('RMSE of linear regression: ', round(ln_rmse, 2))
        print('MAE of linear regression:  ', round(ln_mae, 2))
        print('R2 of linear regression:   ', round(ln_r2, 2))

        # Return evaluation metrics as variables
        return {
            'ln_rmse': ln_rmse,
        }
    
    elif model == 'cb':
        
         # Fit model and predict y-values
        pipe.fit(X_train, y_train)
        y_pred_cb = cat_pipe.predict(X_test)

        # Compute model metrics
        cb_accuracy = accuracy_score(y_test, y_pred_cb)
        cb_recall = recall_score(y_test, y_pred_cb)
        cb_precision = precision_score(y_test, y_pred_cb)
        cb_auc = roc_auc_score(y_test, y_pred_cb)

        # Print results
        print('Accuracy score of CatBoost classifier: ', round(cb_accuracy, 2))
        print('Recall score of CatBoost classifier: ', round(cb_recall, 2))
        print('Precision score of CatBoost classifier: ', round(cb_precision, 2))
        print('AUC score of CatBoost classifier: ', round(cb_auc, 2))

        # Return evaluation metrics as variables
        return {
            'cb_accuracy': cb_accuracy,
            'cb_recall': cb_recall,
            'cb_precision': cb_precision,
            'cb_auc': cb_auc
        }

    else:
        print('Please specify the type of model you want to evaluate.')

print('Performance of default XGBoost model:')
evaluate_model('xgb', xgb_pipe, X_train, y_train, X_test, y_test)


# %%
# TUNE XGBOOST HYPERPARAMETERS =================================================
# Define the parameter grid
# param_grid = {
#     'xgb_classification__n_estimators': [100, 200, 300, 400, 500, 1000],
#     'xgb_classification__max_depth': [2, 3, 5, 7, 9, 10],
#     'xgb_classification__learning_rate': [0.01, 0.02, 0.03, 0.05, 0.1, 0.2, 0.3],
#     'xgb_classification__subsample': [0.5, 0.75, 1.0],
#     'xgb_classification__colsample_bytree': [0.5, 0.75, 1.0],
#     'xgb_classification__gamma': [0.1, 0.5, 1, 3],
#     'xgb_classification__reg_alpha': [0.1, 0.3, 0.5, 0.8, 1, 10],
#     'xgb_classification__reg_lambda': [0.1, 0.3, 0.5, 0.8, 1, 10]
# }

# # Create GridSearchCV object
# random_search = RandomizedSearchCV(
#     estimator=xgb_pipe,
#     param_distributions=param_grid,
#     scoring='roc_auc',
#     cv=5,
#     n_jobs=-1,
#     verbose=2
# )

# # Fit on training data
# random_search.fit(X_train, y_train)

# print("Best XGBoost parameters:             ", random_search.best_params_)
# print("Best XGBoost cross-validation score: ", random_search.best_score_)

# # Evaluate on test set
# best_xgb_model = random_search.best_estimator_
# y_pred_best_xgb = best_xgb_model.predict(X_test)

# print('Performance of XGBoost model with tuned hyperparameters:')
# evaluate_model('xgb', best_xgb_model, X_train, y_train, X_test, y_test)


# %%
# TRAIN REGRESSION MODEL =======================================================
# Split data again but this time with numeric target
X_train, X_test, y_train, y_test = train_test_split(
    features, target_num, test_size=0.3, random_state=RSEED)

# Add XGBoost prediction as column to X train and X test data
X_train['xgb_pred'] = best_xgb_model.predict(X_train)
X_test['xgb_pred'] = best_xgb_model.predict(X_test)

# Create pipeline for linear model
ln_pipe = Pipeline([
    ('preprocessor', preprocessor_pipe),
    ('linear_regression', LinearRegression())
])

# Train model
ln_pipe.fit(X_train, y_train)

print('Combined performance of tuned XGBoost + linear regression:')
evaluate_model('ln', ln_pipe, X_train, y_train, X_test, y_test)


# END OF MAIN ANALYIS PART =====================================================
# Final note:
# Comparing RMSE and AUC for the linear model with/ -out predicted XGBoost
# values didn't indicate any performance improvement.