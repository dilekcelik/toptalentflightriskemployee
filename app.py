import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from openai import OpenAI

# Set layout
st.set_page_config(layout="wide")
st.title("📊 HR Analytics Dashboard (with GPT-4 Commentary)")

# ✅ OpenAI client (ensure API key is in .streamlit/secrets.toml)
client = OpenAI(api_key=st.secrets["openai_api_key"])

def ai_commentary(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a data analyst providing concise insights on HR data visualizations."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=150
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Error generating commentary: {str(e)}"

# Sidebar
uploaded_file = st.sidebar.file_uploader("Upload HR Data CSV", type=["csv"])
use_ai = st.sidebar.checkbox("💬 Enable AI Commentary", value=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # KPIs
    total_employees = df.shape[0]
    attrition_rate = df['left'].mean() * 100
    avg_satisfaction = df['satisfaction_level'].mean()
    avg_eval = df['last_evaluation'].mean()
    avg_projects = df['number_project'].mean()

    st.markdown("### 📌 Key Metrics")
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("👥 Employees", f"{total_employees:,}")
    kpi2.metric("❌ Attrition Rate", f"{attrition_rate:.2f}%")
    kpi3.metric("😊 Avg. Satisfaction", f"{avg_satisfaction:.2f}")
    kpi4.metric("📈 Avg. Evaluation", f"{avg_eval:.2f}")
    kpi5.metric("📊 Avg. Projects", f"{avg_projects:.1f}")

    if use_ai:
        prompt = f"The HR dataset has {total_employees} employees. Attrition rate is {attrition_rate:.2f}%. Avg satisfaction is {avg_satisfaction:.2f}, evaluation {avg_eval:.2f}, and project count {avg_projects:.1f}. Provide key insights."
        st.markdown(f"💡 **AI Insight:** {ai_commentary(prompt)}")

    st.markdown("---")
    st.subheader("📊 Interactive Visualizations")

    # 1. Satisfaction Distribution
    fig1 = px.histogram(df, x='satisfaction_level', nbins=20, title="Satisfaction Level Distribution",
                        marginal="rug", color_discrete_sequence=['#636EFA'])
    st.plotly_chart(fig1)
    if use_ai:
        prompt = "Explain trends and patterns in the satisfaction level distribution histogram of employees."
        st.markdown(f"💡 **AI Insight:** {ai_commentary(prompt)}")

    # 2. Scatter: Satisfaction vs Evaluation
    fig2 = px.scatter(df, x='satisfaction_level', y='last_evaluation',
                      color=df['left'].map({0: 'Stayed', 1: 'Left'}),
                      title="Satisfaction vs Last Evaluation", labels={"color": "Attrition"},
                      hover_data=['number_project', 'salary'],
                      color_discrete_map={'Stayed': 'green', 'Left': 'red'})
    st.plotly_chart(fig2)
    if use_ai:
        prompt = "Analyze the relationship between satisfaction level and last evaluation using a scatterplot with attrition status."
        st.markdown(f"💡 **AI Insight:** {ai_commentary(prompt)}")

    # 3. Department vs Attrition
    fig3 = px.histogram(df, x='Department', color=df['left'].map({0: 'Stayed', 1: 'Left'}),
                        title="Attrition by Department", barmode='group',
                        labels={"color": "Attrition"},
                        color_discrete_map={'Stayed': 'blue', 'Left': 'orange'})
    st.plotly_chart(fig3)
    if use_ai:
        prompt = "Interpret the bar chart showing attrition by department in the HR dataset."
        st.markdown(f"💡 **AI Insight:** {ai_commentary(prompt)}")

    # 4. Salary vs Attrition
    fig4 = px.histogram(df, x='salary', color=df['left'].map({0: 'Stayed', 1: 'Left'}),
                        title="Attrition by Salary Level", barmode='group',
                        category_orders={"salary": ["low", "medium", "high"]},
                        labels={"color": "Attrition"},
                        color_discrete_map={'Stayed': 'blue', 'Left': 'orange'})
    st.plotly_chart(fig4)
    if use_ai:
        prompt = "Analyze attrition patterns across different salary levels in the HR data bar chart."
        st.markdown(f"💡 **AI Insight:** {ai_commentary(prompt)}")

    # 5. Time Spent vs Attrition
    fig5 = px.box(df, x='left', y='time_spend_company',
                  color=df['left'].map({0: 'Stayed', 1: 'Left'}),
                  title="Time Spent at Company vs Attrition",
                  labels={'left': 'Attrition'},
                  color_discrete_map={'Stayed': 'blue', 'Left': 'orange'})
    st.plotly_chart(fig5)
    if use_ai:
        prompt = "Describe the relationship between time spent at the company and attrition status in the boxplot."
        st.markdown(f"💡 **AI Insight:** {ai_commentary(prompt)}")

    # 6. Promotion vs Attrition
    promo_counts = df.groupby(['promotion_last_5years', 'left']).size().unstack(fill_value=0)
    promo_fig = go.Figure(data=[
        go.Bar(name='Stayed', x=[str(i) for i in promo_counts.index], y=promo_counts[0], marker_color='blue'),
        go.Bar(name='Left', x=[str(i) for i in promo_counts.index], y=promo_counts[1], marker_color='orange')
    ])
    promo_fig.update_layout(title="Promotions in Last 5 Years vs Attrition", barmode='group', xaxis_title="Promoted")
    st.plotly_chart(promo_fig)
    if use_ai:
        prompt = "Analyze how receiving a promotion in the last 5 years is related to employee attrition."
        st.markdown(f"💡 **AI Insight:** {ai_commentary(prompt)}")

    # 7. Work Accident vs Attrition
    acc_counts = df.groupby(['Work_accident', 'left']).size().unstack(fill_value=0)
    acc_fig = go.Figure(data=[
        go.Bar(name='Stayed', x=[str(i) for i in acc_counts.index], y=acc_counts[0], marker_color='green'),
        go.Bar(name='Left', x=[str(i) for i in acc_counts.index], y=acc_counts[1], marker_color='red')
    ])
    acc_fig.update_layout(title="Work Accident vs Attrition", barmode='group', xaxis_title="Had Work Accident")
    st.plotly_chart(acc_fig)
    if use_ai:
        prompt = "What does the data say about the correlation between work accidents and employee attrition?"
        st.markdown(f"💡 **AI Insight:** {ai_commentary(prompt)}")

    # 8. Correlation Heatmap
    corr = df.corr(numeric_only=True)
    heatmap_fig = go.Figure(
        data=go.Heatmap(
            z=corr.values,
            x=corr.columns,
            y=corr.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            colorbar_title="Correlation"
        )
    )
    heatmap_fig.update_layout(title="Correlation Heatmap", xaxis_showgrid=False, yaxis_showgrid=False)
    st.plotly_chart(heatmap_fig)
    if use_ai:
        prompt = "Explain the most important correlations visible in this HR dataset heatmap."
        st.markdown(f"💡 **AI Insight:** {ai_commentary(prompt)}")

else:
    st.info("👈 Please upload a CSV file to begin analysis.")
