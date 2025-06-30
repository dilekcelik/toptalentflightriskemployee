import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(layout="wide")
st.title("\U0001F4CA HR Analytics Dashboard (Styled Version)")

# Upload CSV
uploaded_file = st.sidebar.file_uploader("Upload HR Data CSV", type=["csv"])

# Plotly color palette (qualitative)
color_palette = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', 
                 '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52']

# CSS styles for KPI cards
st.markdown("""
<style>
.kpi-card {
    background-color: #ffffff;
    padding: 15px 20px;
    border-radius: 12px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    text-align: center;
    border-left: 8px solid #636EFA; /* default blue */
}
.kpi-title {
    font-weight: bold;
    font-size: 16px;
    color: #555;
}
.kpi-value {
    font-size: 28px;
    font-weight: 600;
    margin-top: 5px;
    color: #000;
}
.red-border { border-left-color: #EF553B; }
.orange-border { border-left-color: #FFA15A; }
teal-border { border-left-color: #00CC96; }
</style>
""", unsafe_allow_html=True)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    # Metrics
    total_employees = df.shape[0]
    attrition_rate = df['left'].mean() * 100
    avg_satisfaction = df['satisfaction_level'].mean()
    avg_eval = df['last_evaluation'].mean()
    avg_projects = df['number_project'].mean()

    # KPI Cards Section
    st.markdown("### \U0001F4CC Key Metrics")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown(f"""
        <div class="kpi-card teal-border">
            <div class="kpi-title">\U0001F465 Employees</div>
            <div class="kpi-value">{total_employees:,}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="kpi-card red-border">
            <div class="kpi-title">❌ Attrition Rate</div>
            <div class="kpi-value">{attrition_rate:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="kpi-card orange-border">
            <div class="kpi-title">☺ Avg. Satisfaction</div>
            <div class="kpi-value">{avg_satisfaction:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="kpi-card teal-border">
            <div class="kpi-title">\U0001F4C8 Avg. Evaluation</div>
            <div class="kpi-value">{avg_eval:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="kpi-card orange-border">
            <div class="kpi-title">\U0001F4CA Avg. Projects</div>
            <div class="kpi-value">{avg_projects:.1f}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.subheader("\U0001F4CA Interactive Visualizations")

    # Histogram - Satisfaction Level
    st.plotly_chart(
        px.histogram(df, x='satisfaction_level', nbins=20,
                     title="Satisfaction Level Distribution",
                     marginal="rug",
                     color_discrete_sequence=[color_palette[0]])
    )

    # Scatterplot - Satisfaction vs Evaluation
    st.plotly_chart(
        px.scatter(df, x='satisfaction_level', y='last_evaluation',
                   color=df['left'].map({0: 'Stayed', 1: 'Left'}),
                   title="Satisfaction vs Last Evaluation",
                   labels={"color": "Attrition"},
                   hover_data=['number_project', 'salary'],
                   color_discrete_map={'Stayed': color_palette[2], 'Left': color_palette[1]})
    )

    # Histogram - Attrition by Department
    st.plotly_chart(
        px.histogram(df, x='Department',
                     color=df['left'].map({0: 'Stayed', 1: 'Left'}),
                     title="Attrition by Department", barmode='group',
                     labels={"color": "Attrition"},
                     color_discrete_map={'Stayed': color_palette[2], 'Left': color_palette[1]})
    )

    # Histogram - Salary vs Attrition
    st.plotly_chart(
        px.histogram(df, x='salary',
                     color=df['left'].map({0: 'Stayed', 1: 'Left'}),
                     title="Attrition by Salary Level", barmode='group',
                     category_orders={"salary": ["low", "medium", "high"]},
                     labels={"color": "Attrition"},
                     color_discrete_map={'Stayed': color_palette[4], 'Left': color_palette[1]})
    )

    # Boxplot - Time at Company vs Attrition
    st.plotly_chart(
        px.box(df, x='left', y='time_spend_company',
               color=df['left'].map({0: 'Stayed', 1: 'Left'}),
               title="Time Spent at Company vs Attrition",
               labels={'left': 'Attrition'},
               color_discrete_map={'Stayed': color_palette[2], 'Left': color_palette[1]})
    )

    # Promotions and Work Accidents
    promo_counts = df.groupby(['promotion_last_5years', 'left']).size().unstack(fill_value=0)
    acc_counts = df.groupby(['Work_accident', 'left']).size().unstack(fill_value=0)

    promo_fig = go.Figure(data=[
        go.Bar(name='Stayed', x=[str(i) for i in promo_counts.index], y=promo_counts[0], marker_color=color_palette[2]),
        go.Bar(name='Left', x=[str(i) for i in promo_counts.index], y=promo_counts[1], marker_color=color_palette[1])
    ])
    promo_fig.update_layout(title="Promotions in Last 5 Years vs Attrition", barmode='group', xaxis_title="Promoted")

    acc_fig = go.Figure(data=[
        go.Bar(name='Stayed', x=[str(i) for i in acc_counts.index], y=acc_counts[0], marker_color=color_palette[2]),
        go.Bar(name='Left', x=[str(i) for i in acc_counts.index], y=acc_counts[1], marker_color=color_palette[1])
    ])
    acc_fig.update_layout(title="Work Accident vs Attrition", barmode='group', xaxis_title="Had Work Accident")

    st.plotly_chart(promo_fig)
    st.plotly_chart(acc_fig)

    # Correlation Heatmap
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
    st.info("\U0001F448 Please upload a CSV file to begin analysis.")
