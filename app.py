import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configure layout
st.set_page_config(layout="wide")
st.title("📊 HR Analytics Dashboard (Interactive Plotly Version)")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload HR Data CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Metrics
    total_employees = df.shape[0]
    attrition_rate = df['left'].mean() * 100
    avg_satisfaction = df['satisfaction_level'].mean()
    avg_eval = df['last_evaluation'].mean()
    avg_projects = df['number_project'].mean()

    # KPI Cards in a styled container
st.markdown("""
    <div style="background-color:#f0f2f6; padding: 20px; border-radius: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.1);">
        <h3>📌 Key Metrics</h3>
    </div>
""", unsafe_allow_html=True)

kpi_container = st.container()
with kpi_container:
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("👥 Employees", f"{total_employees:,}")
    kpi2.metric("❌ Attrition Rate", f"{attrition_rate:.2f}%")
    kpi3.metric("😊 Avg. Satisfaction", f"{avg_satisfaction:.2f}")
    kpi4.metric("📈 Avg. Evaluation", f"{avg_eval:.2f}")
    kpi5.metric("📊 Avg. Projects", f"{avg_projects:.1f}")

    st.markdown("---")
    st.subheader("📊 Interactive Visualizations")

    # Satisfaction Distribution
    st.plotly_chart(
        px.histogram(df, x='satisfaction_level', nbins=20, title="Satisfaction Level Distribution",
                     marginal="rug", color_discrete_sequence=['#636EFA'])
    )

    # Scatterplot: Satisfaction vs Evaluation
    st.plotly_chart(
        px.scatter(df, x='satisfaction_level', y='last_evaluation', color=df['left'].map({0: 'Stayed', 1: 'Left'}),
                   title="Satisfaction vs Last Evaluation", labels={"color": "Attrition"},
                   hover_data=['number_project', 'salary'], color_discrete_map={'Stayed': 'green', 'Left': 'red'})
    )

    # Department-wise Attrition
    st.plotly_chart(
        px.histogram(df, x='Department', color=df['left'].map({0: 'Stayed', 1: 'Left'}),
                     title="Attrition by Department", barmode='group',
                     labels={"color": "Attrition"}, color_discrete_map={'Stayed': 'blue', 'Left': 'orange'})
    )

    # Salary vs Attrition
    st.plotly_chart(
        px.histogram(df, x='salary', color=df['left'].map({0: 'Stayed', 1: 'Left'}),
                     title="Attrition by Salary Level", barmode='group',
                     category_orders={"salary": ["low", "medium", "high"]},
                     labels={"color": "Attrition"}, color_discrete_map={'Stayed': 'blue', 'Left': 'orange'})
    )

    # Time Spent at Company
    st.plotly_chart(
        px.box(df, x='left', y='time_spend_company',
               color=df['left'].map({0: 'Stayed', 1: 'Left'}),
               title="Time Spent at Company vs Attrition",
               labels={'left': 'Attrition'}, color_discrete_map={'Stayed': 'blue', 'Left': 'orange'})
    )

    # Promotion and Work Accident vs Attrition using Graph Objects
    promo_counts = df.groupby(['promotion_last_5years', 'left']).size().unstack(fill_value=0)
    acc_counts = df.groupby(['Work_accident', 'left']).size().unstack(fill_value=0)

    promo_fig = go.Figure(data=[
        go.Bar(name='Stayed', x=[str(i) for i in promo_counts.index], y=promo_counts[0], marker_color='blue'),
        go.Bar(name='Left', x=[str(i) for i in promo_counts.index], y=promo_counts[1], marker_color='orange')
    ])
    promo_fig.update_layout(title="Promotions in Last 5 Years vs Attrition", barmode='group', xaxis_title="Promoted")

    acc_fig = go.Figure(data=[
        go.Bar(name='Stayed', x=[str(i) for i in acc_counts.index], y=acc_counts[0], marker_color='green'),
        go.Bar(name='Left', x=[str(i) for i in acc_counts.index], y=acc_counts[1], marker_color='red')
    ])
    acc_fig.update_layout(title="Work Accident vs Attrition", barmode='group', xaxis_title="Had Work Accident")

    st.plotly_chart(promo_fig)
    st.plotly_chart(acc_fig)

    # Correlation Heatmap using Graph Objects
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

else:
    st.info("👈 Please upload a CSV file to begin analysis.")
