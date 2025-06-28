import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score, precision_score, f1_score, recall_score

# App title
st.title('Employee Retention Prediction App')

# Sidebar for user input
def app():
    '''
    Streamlit UI for predicting whether an employee will leave or not based on the XGB model.
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

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        sat_lvl = st.number_input('Satisfaction Level', min_value=0.0, max_value=1.0, value=0.5)
        tenure = st.number_input('Tenure (Years)', min_value=0, max_value=100, value=2)
    with col2:
        last_eval = st.number_input('Last Evaluation Score', min_value=0.0, max_value=1.0, value=0.5)
        dept = st.selectbox('Department', departments.keys())
    with col3:
        num_proj = st.number_input('Number of Projects', min_value=2, max_value=7, value=4)
        sal = st.selectbox('Salary Level', salaries.keys(), index=1)
    with col4:
        ave_hours = st.number_input('Average Monthly Hours', min_value=0, max_value=744, value=180)
        work_acc = st.checkbox('Had Work Accident?')
        promo = st.checkbox('Promoted in Last 5 Years?')

    # Build input DataFrame
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

    # One-hot encode department and salary
    user_df[departments[dept]] = 1
    user_df[salaries[sal]] = 1

    return user_df

#####################
# Load and prepare data
def load_data(filepath):
    df = pd.read_csv(filepath)
    df.rename(columns={
        'average_montly_hours': 'average_monthly_hours',
        'time_spend_company': 'tenure',
        'Work_accident': 'work_accident',
        'Department': 'department'
    }, inplace=True)
    df.drop_duplicates(inplace=True)
    return df

def split_data(df):
    df_dummies = pd.get_dummies(df)
    y = df_dummies['left']
    X = df_dummies.drop('left', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
    return df_dummies, X_train, X_test, y_train, y_test

#####################
# Train XGBoost Model
def get_xgb(X_train, X_test, y_train):
    xgb = XGBClassifier(
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=100,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    xgb.fit(X_train, y_train)
    y_pred = xgb.predict(X_test)
    return xgb, y_pred

#####################
# Main logic

# Load dataset
df = load_data('HR_Analytics.csv')
df_dummies, X_train, X_test, y_train, y_test = split_data(df)

# Train model
xgb, y_pred = get_xgb(X_train, X_test, y_train)

# Get user input
user_df = app()

# Prediction Button
if st.button("Predict Employee Retention"):
    # Match columns
    user_df = user_df.reindex(columns=X_train.columns, fill_value=0)
    prediction = xgb.predict(user_df)[0]
    if prediction == 1:
        st.warning("⚠️ The employee is likely to leave the company.")
    else:
        st.success("✅ The employee is likely to stay at the company.")
