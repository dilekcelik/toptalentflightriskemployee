import streamlit as st
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Title
st.title("HR Analytics Dashboard with XGBoost Prediction")

# Upload Data
st.sidebar.header("Upload HR Data CSV")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])

# Load dataset
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Dataset Preview")
    st.write(df.head())

    # Encode categorical features
    df_encoded = df.copy()
    label_encoders = {}
    for col in ['Department', 'salary']:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Split
    X = df_encoded.drop("left", axis=1)
    y = df_encoded["left"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load or train model
    try:
        model = joblib.load("xgb_model.pkl")
        st.success("Loaded pretrained XGBoost model.")
    except:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        joblib.dump(model, "xgb_model.pkl")
        st.warning("Trained and saved new XGBoost model.")

    # Predictions on test set
    y_pred = model.predict(X_test)
    st.subheader("Prediction on Test Set")
    st.write(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).head())

    # Form for new input
    st.sidebar.header("Predict for a New Employee")

    def user_input_features():
        satisfaction_level = st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5)
        last_evaluation = st.sidebar.slider('Last Evaluation', 0.0, 1.0, 0.5)
        number_project = st.sidebar.slider('Number of Projects', 1, 10, 3)
        average_montly_hours = st.sidebar.slider('Average Monthly Hours', 50, 310, 160)
        time_spend_company = st.sidebar.slider('Time Spent at Company (Years)', 1, 10, 3)
        Work_accident = st.sidebar.selectbox('Work Accident', [0, 1])
        promotion_last_5years = st.sidebar.selectbox('Promotion in Last 5 Years', [0, 1])
        Department = st.sidebar.selectbox('Department', df['Department'].unique())
        salary = st.sidebar.selectbox('Salary Level', df['salary'].unique())

        # Encode
        dept_encoded = label_encoders['Department'].transform([Department])[0]
        salary_encoded = label_encoders['salary'].transform([salary])[0]

        data = {
            'satisfaction_level': satisfaction_level,
            'last_evaluation': last_evaluation,
            'number_project': number_project,
            'average_montly_hours': average_montly_hours,
            'time_spend_company': time_spend_company,
            'Work_accident': Work_accident,
            'promotion_last_5years': promotion_last_5years,
            'Department': dept_encoded,
            'salary': salary_encoded
        }
        return pd.DataFrame(data, index=[0])

    input_df = user_input_features()

    if st.sidebar.button("Predict"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(f"🔴 This employee is likely to leave. (Probability: {proba:.2f})")
        else:
            st.success(f"🟢 This employee is likely to stay. (Probability of leaving: {proba:.2f})")

    # Visualizations
    st.subheader("Exploratory Data Analysis")
    st.markdown("### Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(df_encoded.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    st.markdown("### Distribution of Employee Satisfaction")
    fig2, ax2 = plt.subplots()
    sns.histplot(df['satisfaction_level'], kde=True, ax=ax2)
    st.pyplot(fig2)

else:
    st.info("👈 Please upload a CSV file to get started.")
