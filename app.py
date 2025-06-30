import streamlit as st
import pandas as pd
import xgboost as xgb
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import openai

# OpenAI 
openai.api_key = st.secrets["openai_api_key"]

def generate_commentary(prompt):
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst who provides concise and insightful commentary."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=150
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"❌ Error generating AI commentary: {str(e)}"

st.title("🩺 Health Care Workforce Analytics with AI Commentary")

# Upload CSV
st.sidebar.header("Upload HR Data CSV")
uploaded_file = st.sidebar.file_uploader("Choose a file", type=["csv"])
use_ai = st.sidebar.checkbox("💬 Enable AI Commentary", value=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Preview")
    st.write(df.head())

    # Encode categorical features
    df_encoded = df.copy()
    label_encoders = {}
    for col in ['Department', 'salary']:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col])
        label_encoders[col] = le

    # Model prep
    X = df_encoded.drop("left", axis=1)
    y = df_encoded["left"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Load or train model
    try:
        model = joblib.load("xgb_model.pkl")
        st.success("✅ Loaded pretrained XGBoost model.")
    except:
        model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train, y_train)
        joblib.dump(model, "xgb_model.pkl")
        st.warning("⚠️ Trained and saved new XGBoost model.")

    # Prediction preview
    y_pred = model.predict(X_test)
    st.subheader("🔍 Prediction Sample")
    st.write(pd.DataFrame({"Actual": y_test.values, "Predicted": y_pred}).head())

    # New employee prediction
    st.sidebar.header("🧠 Predict a New Employee")
    def user_input_features():
        satisfaction_level = st.sidebar.slider('Satisfaction Level', 0.0, 1.0, 0.5)
        last_evaluation = st.sidebar.slider('Last Evaluation', 0.0, 1.0, 0.5)
        number_project = st.sidebar.slider('Number of Projects', 1, 10, 3)
        average_montly_hours = st.sidebar.slider('Avg. Monthly Hours', 50, 310, 160)
        time_spend_company = st.sidebar.slider('Years at Company', 1, 10, 3)
        Work_accident = st.sidebar.selectbox('Work Accident', [0, 1])
        promotion_last_5years = st.sidebar.selectbox('Promotion (Last 5 Yrs)', [0, 1])
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

    if st.sidebar.button("🚀 Predict"):
        prediction = model.predict(input_df)[0]
        proba = model.predict_proba(input_df)[0][1]

        st.subheader("📈 Prediction Result")
        if prediction == 1:
            st.error(f"🔴 This employee is likely to leave. (Probability: {proba:.2f})")
        else:
            st.success(f"🟢 This employee is likely to stay. (Leaving Probability: {proba:.2f})")

    # Visualizations with commentary
    st.subheader("📉 Visual Analytics")

    # 1. Satisfaction Level
    fig1 = px.histogram(df, x="satisfaction_level", nbins=30, title="Employee Satisfaction Distribution")
    st.plotly_chart(fig1)
    if use_ai:
        prompt = "The histogram shows employee satisfaction levels from a healthcare organization. Describe the insights and trends."
        st.markdown(f"💡 **AI Insight:** {generate_commentary(prompt)}")

    # 2. Average Monthly Hours
    fig2 = px.box(df, y="average_montly_hours", title="Average Monthly Working Hours")
    st.plotly_chart(fig2)
    if use_ai:
        prompt = "The boxplot shows the distribution of average monthly working hours. What does it reveal about workload?"
        st.markdown(f"💡 **AI Insight:** {generate_commentary(prompt)}")

    # 3. Time at Company
    fig3 = px.histogram(df, x="time_spend_company", color="left", barmode="group", title="Time at Company vs Turnover")
    st.plotly_chart(fig3)
    if use_ai:
        prompt = "The grouped histogram compares time spent at company with whether employees left. Provide insights."
        st.markdown(f"💡 **AI Insight:** {generate_commentary(prompt)}")

    # 4. Department vs Turnover
    fig4 = px.histogram(df, x="Department", color="left", barmode="group", title="Turnover by Department")
    st.plotly_chart(fig4)
    if use_ai:
        prompt = "The bar chart shows employee turnover across departments. What trends are visible?"
        st.markdown(f"💡 **AI Insight:** {generate_commentary(prompt)}")

    # 5. Salary vs Turnover
    fig5 = px.histogram(df, x="salary", color="left", barmode="group", title="Turnover by Salary Level")
    st.plotly_chart(fig5)
    if use_ai:
        prompt = "The bar chart illustrates turnover by salary level. Describe any patterns or risks."
        st.markdown(f"💡 **AI Insight:** {generate_commentary(prompt)}")

else:
    st.info("📥 Please upload a CSV file to get started.")
