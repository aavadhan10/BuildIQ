import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import plotly.express as px
from datetime import datetime, timedelta

# Set page config with a more professional look
st.set_page_config(
    page_title="BuildIQ Demo",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar with company logo placeholder and navigation
st.sidebar.title("🏢 BuildIQ")
st.sidebar.markdown("---")
selected_feature = st.sidebar.radio(
    "Select Feature",
    ["Tower Placement Map", "Churn Risk Alerts", "ROI Calculator"]
)

# Sample data generation functions
def generate_tower_data():
    return pd.DataFrame({
        'latitude': np.random.uniform(37.7, 37.9, 10),
        'longitude': np.random.uniform(-122.5, -122.3, 10),
        'priority_score': np.random.uniform(0, 1, 10),
        'population_density': np.random.uniform(1000, 5000, 10),
        'existing_coverage': np.random.uniform(0.3, 0.9, 10),
        'expected_roi': np.random.uniform(0.1, 0.3, 10)
    })

def generate_churn_data():
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    base_risk = np.sin(np.linspace(0, 4*np.pi, len(dates))) * 0.3 + 0.5
    noise = np.random.normal(0, 0.1, len(dates))
    return pd.DataFrame({
        'date': dates,
        'churn_risk': np.clip(base_risk + noise, 0, 1),
        'network_quality': np.random.uniform(0.5, 1, len(dates)),
        'customer_complaints': np.random.randint(0, 100, len(dates)),
        'revenue_impact': np.random.uniform(10000, 50000, len(dates))
    })

def show_tower_placement():
    st.title("🗼 Tower Placement Map")
    st.write("Interactive map showing optimal locations for new tower placement based on ML analysis")
    
    # Generate sample data
    tower_data = generate_tower_data()
    
    # Create map centered on San Francisco
    m = folium.Map(location=[37.8, -122.4], zoom_start=12)
    
    # Add markers for potential tower locations
    for idx, row in tower_data.iterrows():
        # Color coding based on priority score
        color = 'red' if row['priority_score'] > 0.7 else 'orange' if row['priority_score'] > 0.4 else 'green'
        
        # Create detailed popup content
        popup_content = f"""
            <div style='font-family: Arial'>
                <b>Location {idx + 1}</b><br>
                Priority: {row['priority_score']:.2f}<br>
                Population: {row['population_density']:.0f}<br>
                Coverage: {row['existing_coverage']:.2%}<br>
                Expected ROI: {row['expected_roi']:.1%}
            </div>
        """
        
        # Add marker to map
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=10,
            color=color,
            popup=folium.Popup(popup_content, max_width=200),
            fill=True
        ).add_to(m)
    
    # Display map
    folium_static(m)
    
    # Display key metrics in columns
    st.markdown("### Key Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("High Priority Locations", len(tower_data[tower_data['priority_score'] > 0.7]))
    with col2:
        st.metric("Avg Population Density", f"{tower_data['population_density'].mean():.0f}")
    with col3:
        st.metric("Coverage Gap", f"{(1 - tower_data['existing_coverage'].mean()):.1%}")
    with col4:
        st.metric("Expected ROI", f"{tower_data['expected_roi'].mean():.1%}")

    # Display data table without matplotlib dependency
    st.markdown("### Detailed Location Analysis")
    st.dataframe(
        tower_data.round(3),
        hide_index=True,
        column_config={
            "priority_score": st.column_config.ProgressColumn(
                "Priority Score",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "expected_roi": st.column_config.NumberColumn(
                "Expected ROI",
                format="%.1%"
            ),
            "existing_coverage": st.column_config.NumberColumn(
                "Existing Coverage",
                format="%.1%"
            )
        }
    )

def show_churn_risk():
    st.title("⚠️ Churn Risk Alerts")
    st.write("Real-time monitoring of network performance and churn risk patterns")
    
    # Generate sample data
    churn_data = generate_churn_data()
    
    # Create time series plot
    fig = px.line(churn_data, x='date', 
                  y=['churn_risk', 'network_quality'],
                  title='Churn Risk vs Network Quality Over Time',
                  labels={'value': 'Score', 'variable': 'Metric'},
                  color_discrete_map={'churn_risk': 'red', 'network_quality': 'green'})
    
    # Update layout for better readability
    fig.update_layout(
        height=400,
        legend_title_text='Metrics',
        xaxis_title='Date',
        yaxis_title='Score'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Alert system based on current risk
    current_risk = churn_data['churn_risk'].iloc[-1]
    if current_risk > 0.7:
        st.error("🚨 High churn risk detected! Immediate action required in identified zones.")
    elif current_risk > 0.4:
        st.warning("⚠️ Moderate churn risk detected. Monitoring situation closely.")
    else:
        st.success("✅ Churn risk levels are currently normal.")
    
    # Display key metrics
    st.markdown("### Real-time Metrics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Current Churn Risk", 
            f"{current_risk:.1%}",
            delta=f"{(current_risk - churn_data['churn_risk'].iloc[-2])*100:.1f}%"
        )
    with col2:
        st.metric(
            "Network Quality", 
            f"{churn_data['network_quality'].iloc[-1]:.1%}"
        )
    with col3:
        st.metric(
            "Active Complaints", 
            churn_data['customer_complaints'].iloc[-1]
        )
    with col4:
        st.metric(
            "Revenue at Risk", 
            f"${churn_data['revenue_impact'].iloc[-1]:,.0f}"
        )

def show_roi_calculator():
    st.title("💰 ROI Calculator")
    st.write("AI-powered investment planning and ROI forecasting system")
    
    # Investment parameters input
    st.markdown("### Investment Parameters")
    col1, col2 = st.columns(2)
    with col1:
        investment_amount = st.number_input(
            "Investment Amount ($)", 
            min_value=100000, 
            value=1000000, 
            step=100000,
            format="%d"
        )
        time_period = st.slider("Investment Timeline (Years)", 1, 5, 3)
    with col2:
        market_type = st.selectbox("Market Type", ["Urban", "Suburban", "Rural"])
        risk_level = st.select_slider(
            "Risk Level", 
            options=["Low", "Medium", "High"],
            value="Medium"
        )
    
    # ROI Calculations
    base_roi = {"Urban": 0.15, "Suburban": 0.12, "Rural": 0.08}[market_type]
    risk_multiplier = {"Low": 0.8, "Medium": 1.0, "High": 1.2}[risk_level]
    roi = base_roi * risk_multiplier
    
    # Market condition adjustment
    market_conditions = st.slider(
        "Market Condition Adjustment", 
        -0.05, 
        0.05, 
        0.0, 
        0.01,
        format="%.2f"
    )
    roi += market_conditions
    
    # Calculate projections
    years = list(range(time_period + 1))
    projections = [investment_amount * (1 + roi) ** year for year in years]
    
    # Create projection chart
    fig = px.line(
        x=years, 
        y=projections,
        labels={'x': 'Year', 'y': 'Projected Value ($)'},
        title='Projected Investment Growth'
    )
    
    # Add conservative estimate
    fig.add_scatter(
        x=years, 
        y=[investment_amount * (1 + roi*0.8) ** year for year in years],
        name='Conservative Estimate',
        line=dict(dash='dash')
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Display summary metrics
    st.markdown("### ROI Summary")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Expected Annual ROI", f"{roi:.1%}")
    with col2:
        st.metric("5-Year Return", f"${projections[-1]:,.0f}")
    with col3:
        st.metric("Total Growth", f"{(projections[-1]/investment_amount - 1):.1%}")
    with col4:
        st.metric("Break-even Time", f"{np.log(2)/np.log(1+roi):.1f} years")

# Main app logic
if selected_feature == "Tower Placement Map":
    show_tower_placement()
elif selected_feature == "Churn Risk Alerts":
    show_churn_risk()
else:
    show_roi_calculator()

# Footer
st.markdown("---")
st.markdown("BuildIQ Demo - Powered by Machine Learning and Generative AI")
