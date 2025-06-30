import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set wide layout
st.set_page_config(layout="wide")
st.title("📊 HR Analytics Dashboard")
st.markdown("### Employee Flight Risk Overview")

# Upload file
uploaded_file = st.sidebar.file_uploader("Upload HR Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.markdown("#### 🔍 Dataset Preview")
    st.dataframe(df.head())

    # Key Metrics
    total_employees = df.shape[0]
    attrition_rate = df['left'].mean() * 100
    avg_satisfaction = df['satisfaction_level'].mean()
    avg_eval = df['last_evaluation'].mean()
    avg_projects = df['number_project'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("👥 Total Employees", f"{total_employees:,}")
    col2.metric("❌ Attrition Rate", f"{attrition_rate:.2f}%")
    col3.metric("😊 Avg. Satisfaction", f"{avg_satisfaction:.2f}")

    col4, col5 = st.columns(2)
    col4.metric("📈 Avg. Evaluation Score", f"{avg_eval:.2f}")
    col5.metric("📊 Avg. Project Count", f"{avg_projects:.1f}")

    st.markdown("---")
    st.subheader("📊 Visualizations")

    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Satisfaction Level Distribution")
        fig, ax = plt.subplots()
        sns.histplot(df['satisfaction_level'], kde=True, ax=ax, bins=20)
        st.pyplot(fig)

    with colB:
        st.markdown("#### Satisfaction vs Last Evaluation")
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x='satisfaction_level', y='last_evaluation', hue='left', alpha=0.7, ax=ax)
        st.pyplot(fig)

    colC, colD = st.columns(2)
    with colC:
        st.markdown("#### Attrition by Department")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='Department', hue='left', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with colD:
        st.markdown("#### Attrition by Salary Level")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='salary', hue='left', ax=ax,
                      order=sorted(df['salary'].unique()))  # Ensures order: low < medium < high
        st.pyplot(fig)

    colE, colF = st.columns(2)
    with colE:
        st.markdown("#### Time Spent at Company vs Attrition")
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x='left', y='time_spend_company', ax=ax)
        st.pyplot(fig)

    with colF:
        st.markdown("#### Promotion & Work Accident vs Attrition")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x='promotion_last_5years', hue='left', ax=ax)
        plt.title("Promotions vs Attrition")
        st.pyplot(fig)

        fig2, ax2 = plt.subplots()
        sns.countplot(data=df, x='Work_accident', hue='left', ax=ax2)
        plt.title("Work Accidents vs Attrition")
        st.pyplot(fig2)

    st.markdown("---")
    st.subheader("🔁 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

else:
    st.info("👈 Please upload a CSV file to start exploring your HR data.")
