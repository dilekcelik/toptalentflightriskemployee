import pandas as pd
import streamlit as st
import sklearn
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score


st.title('Employee Retention Prediction')

################

def app():
    '''
    Streamlit UI for predicting whether a employee will leave or not based on the random forest model.
    '''
    departments = {
        'IT': 'department_IT',	
        'RandD': 'department_RandD',
        'Accounting': 'department_accounting',
        'HR': 'department_hr',
        'Management': 'department_management',
        'Marketing': 'department_marketing',
        'Product Management': 'department_product_mng',
        'Sales': 'department_sales',
        'Support': 'department_support',
        'Technical': 'department_technical',
    }	
    salaries = {
        'Low': 'salary_low',
        'Medium': 'salary_medium',
        'High': 'salary_high',
    }

    col1, col2, col3, col4= st.columns(4)
    with col1:
        sat_lvl = st.number_input('Satisfaction Level', min_value=0.0, max_value=1.0, value=0.5)
        tenure = st.number_input('Tenure', min_value=0, max_value=100, value=2)
    with col2:
        last_eval = st.number_input('Last Evaluation', min_value=0.0, max_value=1.0, value=0.5)
        dept = st.selectbox('Department', departments.keys())
    with col3:
        num_proj = st.number_input('Number Projects', min_value=2, max_value=7, value=4)
        sal = st.selectbox('Salary', salaries.keys(), index=1)
    with col4:
        ave_hours = st.number_input('Average Monthly Hours', min_value=0, max_value=744, value=180)
        work_acc = st.checkbox('Work Accident')
        promo = st.checkbox('Promotion in the Last 5 Years')

    user_df = pd.DataFrame({
        'satisfaction_level': sat_lvl,
        'last_evaluation': last_eval,
        'number_project': num_proj, 
        'average_monthly_hours': ave_hours,
        'tenure': tenure,
        'work_accident': int(work_acc),
        'promotion_last_5years': int(promo),
        'department_IT': 0,
        'department_RandD': 0,
        'department_accounting': 0,
        'department_hr': 0,
        'department_management': 0, 
        'department_marketing': 0,
        'department_product_mng': 0,
        'department_sales': 0,
        'department_support': 0,
        'department_technical': 0,
        'salary_high': 0,
        'salary_low': 0,
        'salary_medium': 0,
    }, index=[0])
    user_df[departments[dept]] = 1
    user_df[salaries[sal]] = 1
app()

################ DATA

def load_data(filepath):
    '''
    Loads the .csv file at filepath, renames columns, and drops duplicates.

    Args:
        filepath (string)
    Returns:
        df (DataFrame): Cleaned DataFrame of .csv data.
    '''
    df = pd.read_csv(filepath)
    new_col_names = {
        'average_montly_hours': 'average_monthly_hours',
        'time_spend_company': 'tenure',
        'Work_accident': 'work_accident',
        'Department': 'department'
    }
    df.rename(columns=new_col_names, inplace=True)
    df = df.drop_duplicates()
    return df

df = load_data('HR_Analytics.csv')

################ SPLIT DATA

def split_data(df):
    '''
    Splits the input data for model training. Target is the left column while everything else
    is a feature.

    Args:
        df (DataFrame)
    Returns:
        df_dummies (DataFrame): Input df with label encoding.
        X_train (DataFrame): 80% of feature data.
        X_test (DataFrame): 20% of feature data reserved for model validation.
        y_train (DataFrame): 80% of taget data.
        y_test (DataFrame): 20% of target data reserved for model validation.
    '''
    df_dummies = pd.get_dummies(df)
    y = df_dummies.left
    X = df_dummies.drop('left', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return df_dummies, X_train, X_test, y_train, y_test

################ XGB MODEL
st.subheader('XGBOOST')

code = '''
xgb= XGBClassifier(
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=xgb_model.best_iteration,
    use_label_encoder=False,
    eval_metric='logloss'
)
'''
st.code(code, language='python')

def get_xgb(df, X_train, X_test, y_train):
    '''
    Trains, fits, and predicts a random forest model.

    Args:
        df (DataFrame)
        X_train (DataFrame)
        X_test (DataFrame)
        y_train (DataFrame)
    Returns:
        xgb
        y_pred (DataFrame)
    '''
    #####
    import xgboost as xgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import learning_curve
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Assuming X, y are your actual data and target variables
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    
    # 2. Set parameters
    params = {
        'objective': 'binary:logistic',    # Binary classification task
        'eval_metric': 'logloss',           # Evaluation metric
        'learning_rate': 0.05,              # Learning rate
        'max_depth': 4,                     # Tree depth to prevent overfitting
        'subsample': 0.8,                   # Use 80% of data for each tree to reduce overfitting
        'colsample_bytree': 0.8,            # Subsample features to avoid overfitting
        'seed': 42,                         # Random seed for reproducibility
        'alpha': 0.1,                       # L1 regularization
        'lambda': 1.0,                      # L2 regularization
    }
    
    # 3. Convert data to DMatrix format for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 4. Train model with early stopping
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    
    # Cross-validation to select the best number of boosting rounds
    cv_results = xgb.cv(params, dtrain, nfold=5, num_boost_round=1000, early_stopping_rounds=20,
                        metrics="logloss", seed=42)
    
    best_num_round = cv_results.shape[0]  # Best number of boosting rounds from cross-validation
    
    # 5. Train final model with the best number of boosting rounds
    xgb_model = xgb.train(params, dtrain, num_boost_round=best_num_round, evals=evals, early_stopping_rounds=20)
    
    # 6. Predict on the test set
    y_pred = (xgb_model.predict(dtest) > 0.5).astype(int)
    
    ######
    
    xgb = XGBClassifier(
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    colsample_bytree=0.8,
    n_estimators=xgb_model.best_iteration,
    use_label_encoder=False,
    eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    return xgb, y_pred

df_dummies, X_train, X_test, y_train, y_test = split_data(df)
xgb, y_pred_xgb = get_xgb(df_dummies, X_train, X_test, y_train)

################ PREDICTION

prediction = xgb.predict(user_df)
if prediction == 1:
    st.info('The employee is likely to leave the company.')
elif prediction == 0:
    st.info('The employee is likely to continue working at the company.')

