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
st.set_page_config(page_title="BuildIQ by Ankita Avadhani", layout="wide", page_icon="🏢")

# Custom CSS
st.markdown("""
    <style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #424242;
        font-style: italic;
    }
    </style>
    """, unsafe_allow_html=True)

# Title and Branding
st.markdown('<p class="main-title">BuildIQ</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">by Ankita Avadhani</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("🏢 BuildIQ by Ankita Avadhani")
page = st.sidebar.radio("Navigate", 
    ["Tower Placement", "Network Pattern Monitor + Risk Alert", "ROI Calculator"])

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

def generate_network_pattern_data():
    # Generate hourly data for the past month
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    
    # Create complex seasonal patterns
    daily_pattern = np.sin(2 * np.pi * (dates.hour / 24))  # Daily cycle
    weekly_pattern = np.sin(2 * np.pi * (dates.dayofweek / 7))  # Weekly cycle
    monthly_pattern = np.sin(2 * np.pi * (dates.day / 30))  # Monthly cycle
    
    # Add growing trend
    trend = np.linspace(0, 0.5, len(dates))
    
    # Combine patterns with different weights
    combined_pattern = (
        0.4 * daily_pattern +  # Strong daily pattern
        0.2 * weekly_pattern +  # Medium weekly pattern
        0.1 * monthly_pattern +  # Light monthly pattern
        0.3 * trend  # Significant upward trend
    )
    
    # Add random noise
    noise = np.random.normal(0, 0.1, len(dates))
    
    # Create the final signal
    network_load = np.clip(combined_pattern + noise, 0, 1)
    
    # Add performance metrics
    latency = 20 + 5 * daily_pattern + 2 * weekly_pattern + np.random.normal(0, 1, len(dates))
    packet_loss = np.clip(0.02 + 0.01 * np.abs(daily_pattern) + np.random.normal(0, 0.005, len(dates)), 0, 0.1)
    
    return pd.DataFrame({
        'timestamp': dates,
        'network_load': network_load,
        'baseline': combined_pattern,
        'trend': trend,
        'daily_seasonal': daily_pattern,
        'weekly_seasonal': weekly_pattern,
        'latency': latency,
        'packet_loss': packet_loss
    })

def show_tower_placement():
    st.title("🗼 Tower Infrastructure Management")
    
    st.subheader("Smart Tower Placement Map")
    st.write("Machine Learning Powered analysis for optimal tower locations")
    
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

def show_network_pattern_monitor():
    st.title("📊 Network Pattern Monitor + Risk Alert")
    st.write("AI-powered analysis to detect and alert on atypical network patterns, accounting for seasonality and trends")
    
    # Generate data
    data = generate_network_pattern_data()
    recent_data = data.tail(48)  # Last 48 hours
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["🔍 Real-time Monitor", "📈 Pattern Analysis", "⚠️ Risk Alerts"])
    
    with tab1:
        st.subheader("Live Network Status")
        
        # Current metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            current_load = data['network_load'].iloc[-1]
            expected_load = data['baseline'].iloc[-1]
            deviation = current_load - expected_load
            st.metric("Network Load", 
                     f"{current_load:.2%}",
                     f"{deviation*100:.1f}% from expected",
                     delta_color="inverse")
        with col2:
            current_latency = data['latency'].iloc[-1]
            st.metric("Latency",
                     f"{current_latency:.1f}ms",
                     f"{current_latency - data['latency'].mean():.1f}ms from avg",
                     delta_color="inverse")
        with col3:
            current_loss = data['packet_loss'].iloc[-1]
            st.metric("Packet Loss",
                     f"{current_loss:.2%}",
                     f"{(current_loss - data['packet_loss'].mean())*100:.2f}% from avg",
                     delta_color="inverse")
        
        # Real-time comparison chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['network_load'],
            name='Actual Load',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['baseline'],
            name='Expected Pattern',
            line=dict(color='green', dash='dash')
        ))
        fig.update_layout(title='Network Load vs Expected Pattern (Last 48 Hours)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Pattern Components")
        
        col1, col2 = st.columns(2)
        with col1:
            # Daily pattern analysis
            daily_avg = data.groupby(data['timestamp'].dt.hour)['network_load'].mean()
            fig_daily = px.line(daily_avg, 
                              title='Daily Network Pattern',
                              labels={'value': 'Average Load', 'index': 'Hour of Day'})
            st.plotly_chart(fig_daily)
            
        with col2:
            # Weekly pattern analysis
            weekly_avg = data.groupby(data['timestamp'].dt.dayofweek)['network_load'].mean()
            fig_weekly = px.line(weekly_avg,
                               title='Weekly Network Pattern',
                               labels={'value': 'Average Load', 'index': 'Day of Week'})
            st.plotly_chart(fig_weekly)
        
        # Trend analysis
        st.subheader("Long-term Trend")
        fig_trend = px.line(data, x='timestamp', y=['trend', 'network_load'],
                           title='Network Load Trend Analysis')
        st.plotly_chart(fig_trend)
    
    with tab3:
        st.subheader("Network Risk Analysis")
        
        # Calculate deviations and identify risks
        data['deviation'] = abs(data['network_load'] - data['baseline'])
        threshold = data['deviation'].mean() + 2 * data['deviation'].std()
        risk_periods = data[data['deviation'] > threshold].copy()
        
        if not risk_periods.empty:
            st.error("🚨 Atypical Network Patterns Detected!")
            
            # Show recent risks
            for _, risk in risk_periods.tail(3).iterrows():
                severity = "High" if risk['deviation'] > threshold * 1.5 else "Medium"
                color = "red" if severity == "High" else "orange"
                st.warning(f"""
                    :{color}[{severity} Risk Pattern] detected at {risk['timestamp'].strftime('%Y-%m-%d %H:%M')}
                    - Expected Load: {risk['baseline']:.2%}
                    - Actual Load: {risk['network_load']:.2%}
                    - Deviation: {risk['deviation']:.2%}
                    
                    Recommended Actions:
                    1. Review network segments for anomalies
                    2. Check for maintenance needs
                    3. Monitor customer experience metrics
                """)
        
        # Risk visualization
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['network_load'],
            name='Network Load',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=risk_periods['timestamp'],
            y=risk_periods['network_load'],
            mode='markers',
            name='Risk Periods',
            marker=dict(color='red', size=10)
        ))
        fig.update_layout(title='Network Load with Risk Periods Highlighted')
        st.plotly_chart(fig)
        
        # Risk metrics
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Risk Periods Detected", 
                     len(risk_periods),
                     f"{len(risk_periods)/len(data)*100:.1f}% of time")
        with col2:
            st.metric("Average Risk Severity",
                     f"{risk_periods['deviation'].mean()/threshold:.1f}x threshold")

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
elif page == "Network Pattern Monitor + Risk Alert":
    show_network_pattern_monitor()
elif page == "ROI Calculator":
    show_roi_calculator()

# Footer
st.markdown("---")
st.markdown("© 2024 BuildIQ by Ankita Avadhani - Powered by Machine Learning and Generative AI")
