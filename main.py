import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import folium
from streamlit_folium import folium_static
from datetime import datetime, timedelta
import time

# Page config
st.set_page_config(page_title="BuildIQ Demo by Ankita Avadhani", layout="wide", page_icon="🏢")

# Sidebar
st.sidebar.title("🏢 BuildIQ Features")
page = st.sidebar.radio("Navigate", 
    ["Tower Placement", "Network Monitor", "Churn Risk", 
     "ROI Calculator", "Maintenance", "Weather Impact"])

# Data generation functions
def generate_tower_data():
    return pd.DataFrame({
        'latitude': np.random.uniform(37.7, 37.9, 10),
        'longitude': np.random.uniform(-122.5, -122.3, 10),
        'priority_score': np.random.uniform(0, 1, 10),
        'population_density': np.random.uniform(1000, 5000, 10),
        'existing_coverage': np.random.uniform(0.3, 0.9, 10),
        'expected_roi': np.random.uniform(0.1, 0.3, 10)
    })

def simulate_network_traffic():
    return {
        'data_volume': np.random.normal(500, 100),
        'active_users': np.random.randint(1000, 5000),
        'network_load': np.random.uniform(0.3, 0.9),
        'latency': np.random.normal(20, 5),
        'packet_loss': np.random.uniform(0, 0.02)
    }

def generate_maintenance_data():
    return pd.DataFrame({
        'Tower_ID': range(1, 6),
        'Last_Maintenance': pd.date_range(start='2023-01-01', periods=5, freq='M'),
        'Health_Score': np.random.uniform(0.5, 1, 5),
        'Days_To_Maintenance': np.random.randint(30, 365, 5),
        'Critical_Components': np.random.choice(['Antenna', 'Power', 'Network', 'Structure'], 5)
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

# Feature implementations
def show_tower_placement():
    st.title("🗼 Tower Placement Map")
    st.write("Interactive map showing optimal locations for new tower placement")
    
    # Generate data
    tower_data = generate_tower_data()
    
    # Create map
    m = folium.Map(location=[37.8, -122.4], zoom_start=12)
    
    # Add markers
    for idx, row in tower_data.iterrows():
        color = 'red' if row['priority_score'] > 0.7 else 'orange' if row['priority_score'] > 0.4 else 'green'
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=10,
            color=color,
            popup=f"""
                <b>Location {idx + 1}</b><br>
                Priority: {row['priority_score']:.2f}<br>
                Population: {row['population_density']:.0f}<br>
                Coverage: {row['existing_coverage']:.2%}<br>
                Expected ROI: {row['expected_roi']:.1%}
            """,
            fill=True
        ).add_to(m)
    
    folium_static(m)
    
    # 3D Coverage Visualization
    st.subheader("3D Coverage Analysis")
    x = np.linspace(-5, 5, 50)
    y = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))

    fig = go.Figure(data=[go.Surface(z=Z, x=x, y=y)])
    fig.update_layout(title='Signal Strength Distribution')
    st.plotly_chart(fig)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("High Priority Locations", len(tower_data[tower_data['priority_score'] > 0.7]))
    with col2:
        st.metric("Avg Population Density", f"{tower_data['population_density'].mean():.0f}")
    with col3:
        st.metric("Coverage Gap", f"{(1 - tower_data['existing_coverage'].mean()):.1%}")
    with col4:
        st.metric("Expected ROI", f"{tower_data['expected_roi'].mean():.1%}")

def show_network_monitor():
    st.title("📊 Live Network Monitor")
    
    # Create placeholder for live metrics
    metrics_placeholder = st.empty()
    chart_placeholder = st.empty()
    
    # Initialize data storage
    if 'network_data' not in st.session_state:
        st.session_state.network_data = []
    
    # Simulate real-time monitoring
    metrics = simulate_network_traffic()
    
    # Update metrics
    with metrics_placeholder.container():
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Network Load", f"{metrics['network_load']:.1%}")
        col2.metric("Active Users", f"{metrics['active_users']:,}")
        col3.metric("Latency", f"{metrics['latency']:.1f}ms")
        col4.metric("Packet Loss", f"{metrics['packet_loss']:.2%}")
    
    # Store and display historical data
    st.session_state.network_data.append(metrics)
    if len(st.session_state.network_data) > 50:
        st.session_state.network_data.pop(0)
    
    # Create historical chart
    df = pd.DataFrame(st.session_state.network_data)
    fig = px.line(df, y=['network_load', 'packet_loss'], 
                 title='Network Performance Trends')
    chart_placeholder.plotly_chart(fig, use_container_width=True)

def show_maintenance_predictions():
    st.title("🔧 Predictive Maintenance")
    
    maintenance_data = generate_maintenance_data()
    
    # Show maintenance schedule
    st.subheader("Tower Maintenance Schedule")
    st.dataframe(
        maintenance_data,
        column_config={
            "Health_Score": st.column_config.ProgressColumn(
                "Tower Health",
                help="Current health score of the tower",
                format="%.2f",
                min_value=0,
                max_value=1,
            ),
            "Days_To_Maintenance": st.column_config.NumberColumn(
                "Days Until Maintenance",
                help="Predicted days until maintenance required",
            )
        }
    )
    
    # Maintenance priority chart
    fig = px.scatter(maintenance_data, 
                    x="Days_To_Maintenance", 
                    y="Health_Score",
                    size="Days_To_Maintenance",
                    color="Critical_Components",
                    hover_data=["Tower_ID"],
                    title="Maintenance Priority Matrix")
    st.plotly_chart(fig)

def show_weather_impact():
    st.title("🌤️ Weather Impact Analysis")
    
    # Generate weather impact data
    weather_data = pd.DataFrame({
        'condition': ['Clear', 'Rain', 'Storm', 'Snow', 'Fog'],
        'signal_strength': [95, 80, 60, 70, 75],
        'affected_towers': np.random.randint(0, 20, 5)
    })
    
    # Weather impact visualization
    fig = px.bar(weather_data, x='condition', y='signal_strength',
                 title='Signal Strength by Weather Condition',
                 labels={'condition': 'Weather', 'signal_strength': 'Signal Strength (%)'})
    st.plotly_chart(fig)
    
    # Real-time weather alerts
    st.subheader("Active Weather Alerts")
    alerts = [
        {"severity": "High", "condition": "Storm", "location": "North Sector"},
        {"severity": "Medium", "condition": "Rain", "location": "West Sector"},
    ]
    
    for alert in alerts:
        color = "red" if alert["severity"] == "High" else "orange"
        st.markdown(f":{color}[{alert['severity']}]: {alert['condition']} in {alert['location']}")

def show_churn_risk():
    st.title("⚠️ Churn Risk Analysis")
    
    # Generate churn data
    churn_data = generate_churn_data()
    
    # Show current risk metrics
    current_risk = churn_data['churn_risk'].iloc[-1]
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current Churn Risk", f"{current_risk:.1%}")
    with col2:
        st.metric("Network Quality", f"{churn_data['network_quality'].iloc[-1]:.1%}")
    with col3:
        st.metric("Active Complaints", churn_data['customer_complaints'].iloc[-1])
    
    # Churn risk over time
    fig = px.line(churn_data, x='date', y=['churn_risk', 'network_quality'],
                 title='Churn Risk vs Network Quality Trends')
    st.plotly_chart(fig)
    
    # Risk alerts
    if current_risk > 0.7:
        st.error("🚨 High churn risk detected! Immediate action required.")
    elif current_risk > 0.4:
        st.warning("⚠️ Moderate churn risk detected. Monitor situation.")

def show_roi_calculator():
    st.title("💰 ROI Calculator")
    
    col1, col2 = st.columns(2)
    with col1:
        investment = st.number_input("Investment Amount ($)", 
                                   min_value=100000, 
                                   value=1000000, 
                                   step=100000)
        market_type = st.selectbox("Market Type", 
                                 ["Urban", "Suburban", "Rural"])
    with col2:
        time_period = st.slider("Investment Timeline (Years)", 1, 5, 3)
        risk_level = st.select_slider("Risk Level", 
                                    options=["Low", "Medium", "High"])
    
    # Calculate ROI
    base_roi = {"Urban": 0.15, "Suburban": 0.12, "Rural": 0.08}[market_type]
    risk_multiplier = {"Low": 0.8, "Medium": 1.0, "High": 1.2}[risk_level]
    roi = base_roi * risk_multiplier
    
    # Generate projections
    years = list(range(time_period + 1))
    projections = [investment * (1 + roi) ** year for year in years]
    
    # Show projections
    fig = px.line(x=years, y=projections,
                 title='Investment Growth Projection',
                 labels={'x': 'Years', 'y': 'Value ($)'})
    st.plotly_chart(fig)
    
    # Show metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Expected Annual ROI", f"{roi:.1%}")
    with col2:
        st.metric("5-Year Return", f"${projections[-1]:,.0f}")
    with col3:
        st.metric("Break-even Time", f"{np.log(2)/np.log(1+roi):.1f} years")

# Main app logic
if page == "Tower Placement":
    show_tower_placement()
elif page == "Network Monitor":
    show_network_monitor()
elif page == "Maintenance":
    show_maintenance_predictions()
elif page == "Weather Impact":
    show_weather_impact()
elif page == "Churn Risk":
    show_churn_risk()
elif page == "ROI Calculator":
    show_roi_calculator()

# Footer
st.markdown("---")
st.markdown("BuildIQ Demo - Powered by Machine Learning and Generative AI")
