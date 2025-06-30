import streamlit as st
import pandas as pd
import plotly.express as px
import openai

# Load OpenAI key
openai.api_key = st.secrets["openai_api_key"]

st.title("HR Analytics Dashboard with AI Insights")

# Load data
df = pd.read_csv("HR_Analytics.csv")

# Column mapping
df.columns = [
    "satisfaction_level", "last_evaluation", "number_project", "average_monthly_hours",
    "time_spend_company", "work_accident", "left", "promotion_last_5years", "department", "salary"
]

# Function to get GPT explanation
def get_ai_comment(plot_desc):
    prompt = f"You're a data analyst. Please give a short and insightful comment on this chart: {plot_desc}"
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You're a helpful data analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    return response["choices"][0]["message"]["content"]

# Plot and AI description
def show_plot_and_comment(fig, desc):
    st.plotly_chart(fig)
    with st.spinner("Generating AI insight..."):
        comment = get_ai_comment(desc)
    st.markdown(f"**💡 AI Comment:** {comment}")

# 1. Histogram of Satisfaction Level
fig1 = px.histogram(df, x="satisfaction_level", nbins=20, title="Distribution of Satisfaction Level")
show_plot_and_comment(fig1, "Histogram showing the distribution of employee satisfaction levels.")

# 2. Boxplot of Monthly Hours by Left Status
fig2 = px.box(df, x="left", y="average_monthly_hours", title="Average Monthly Hours by Turnover Status", labels={"left": "Left Company"})
show_plot_and_comment(fig2, "Boxplot of monthly working hours grouped by whether employees left the company.")

# 3. Scatter plot of Satisfaction vs Last Evaluation
fig3 = px.scatter(df, x="satisfaction_level", y="last_evaluation", color="left", title="Satisfaction vs Evaluation (Turnover)", labels={"left": "Left Company"})
show_plot_and_comment(fig3, "Scatter plot of satisfaction level vs last evaluation, colored by whether employees left.")

# 4. Count plot of Department
fig4 = px.histogram(df, x="department", color="left", barmode="group", title="Employees by Department and Turnover")
show_plot_and_comment(fig4, "Bar chart of employee counts by department, grouped by whether they left.")

# 5. Count plot of Salary
fig5 = px.histogram(df, x="salary", color="left", barmode="group", title="Turnover by Salary Level")
show_plot_and_comment(fig5, "Bar chart of employee salary levels grouped by whether they left.")

st.success("✅ All graphs rendered with AI insights!")
