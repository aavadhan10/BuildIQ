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
st.sidebar.title("🏢 BuildIQ Features")
page = st.sidebar.radio("Navigate", 
    ["Tower Placement", "Network Monitor", "Network Pattern Analysis", "ROI Calculator"])

[Previous tower_placement, network_monitor, and roi_calculator functions remain the same...]

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

def show_network_pattern_analysis():
    st.title("📊 Network Pattern Analysis & Anomaly Detection")
    st.write("""
        AI-powered analysis to solve low forecast accuracy due to high seasonality and trending patterns in network performance data.
        This system separates true anomalies from expected seasonal variations.
    """)
    
    # Generate data
    data = generate_network_pattern_data()
    
    # Pattern Decomposition
    tab1, tab2, tab3, tab4 = st.tabs([
        "Real-time Monitoring", 
        "Pattern Analysis", 
        "Seasonality Impact", 
        "Anomaly Detection"
    ])
    
    with tab1:
        st.subheader("Current Network Status")
        
        # Latest metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            current_load = data['network_load'].iloc[-1]
            expected_load = data['baseline'].iloc[-1]
            st.metric("Network Load", 
                     f"{current_load:.2%}",
                     f"{(current_load - expected_load)*100:.1f}% from expected")
        with col2:
            current_latency = data['latency'].iloc[-1]
            st.metric("Latency",
                     f"{current_latency:.1f}ms",
                     f"{current_latency - data['latency'].iloc[-2]:.1f}ms")
        with col3:
            current_loss = data['packet_loss'].iloc[-1]
            st.metric("Packet Loss",
                     f"{current_loss:.2%}",
                     f"{(current_loss - data['packet_loss'].iloc[-2])*100:.2f}%")
        
        # Real-time pattern comparison
        fig = go.Figure()
        recent_data = data.tail(48)  # Last 48 hours
        
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
        
        fig.update_layout(
            title='Network Load: Last 48 Hours',
            xaxis_title='Time',
            yaxis_title='Network Load',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Pattern Decomposition")
        
        # Show individual components
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['trend'],
                               name='Long-term Trend'))
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['daily_seasonal'],
                               name='Daily Pattern'))
        fig.add_trace(go.Scatter(x=data['timestamp'], y=data['weekly_seasonal'],
                               name='Weekly Pattern'))
        
        fig.update_layout(title='Network Pattern Components')
        st.plotly_chart(fig, use_container_width=True)
        
        # Pattern strength analysis
        pattern_strength = pd.DataFrame({
            'Pattern': ['Daily', 'Weekly', 'Monthly', 'Trend'],
            'Strength': [
                data['daily_seasonal'].std(),
                data['weekly_seasonal'].std(),
                (data['network_load'] - data['trend']).std(),
                data['trend'].std()
            ]
        })
        
        fig_strength = px.bar(pattern_strength,
                            x='Pattern',
                            y='Strength',
                            title='Pattern Strength Analysis')
        st.plotly_chart(fig_strength)
    
    with tab3:
        st.subheader("Seasonality Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Daily patterns
            daily_avg = data.groupby(data['timestamp'].dt.hour)['network_load'].agg(['mean', 'std'])
            fig_daily = go.Figure([
                go.Scatter(
                    x=daily_avg.index,
                    y=daily_avg['mean'],
                    name='Mean Load',
                    line=dict(color='blue'),
                ),
                go.Scatter(
                    x=daily_avg.index,
                    y=daily_avg['mean'] + daily_avg['std'],
                    fill=None,
                    line=dict(color='gray'),
                    name='Upper Bound'
                ),
                go.Scatter(
                    x=daily_avg.index,
                    y=daily_avg['mean'] - daily_avg['std'],
                    fill='tonexty',
                    line=dict(color='gray'),
                    name='Lower Bound'
                )
            ])
            fig_daily.update_layout(title='Daily Pattern Analysis')
            st.plotly_chart(fig_daily)
            
        with col2:
            # Weekly patterns
            weekly_avg = data.groupby(data['timestamp'].dt.dayofweek)['network_load'].agg(['mean', 'std'])
            days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            fig_weekly = go.Figure([
                go.Scatter(
                    x=days,
                    y=weekly_avg['mean'],
                    name='Mean Load',
                    line=dict(color='blue')
                ),
                go.Scatter(
                    x=days,
                    y=weekly_avg['mean'] + weekly_avg['std'],
                    fill=None,
                    line=dict(color='gray'),
                    name='Upper Bound'
                ),
                go.Scatter(
                    x=days,
                    y=weekly_avg['mean'] - weekly_avg['std'],
                    fill='tonexty',
                    line=dict(color='gray'),
                    name='Lower Bound'
                )
            ])
            fig_weekly.update_layout(title='Weekly Pattern Analysis')
            st.plotly_chart(fig_weekly)
    
    with tab4:
        st.subheader("Anomaly Detection")
        
        # Calculate deviations
        data['deviation'] = data['network_load'] - data['baseline']
        threshold = data['deviation'].std() * 2  # 2 sigma threshold
        anomalies = data[abs(data['deviation']) > threshold]
        
        # Plot with anomalies highlighted
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['network_load'],
            name='Network Load',
            line=dict(color='blue')
        ))
        
        # Highlight anomalies
        fig.add_trace(go.Scatter(
            x=anomalies['timestamp'],
            y=anomalies['network_load'],
            mode='markers',
            name='Anomalies',
            marker=dict(color='red', size=10)
        ))
        
        fig.update_layout(title='Network Load with Detected Anomalies')
        st.plotly_chart(fig, use_container_width=True)
        
        # Anomaly statistics
        st.metric("Anomaly Rate", f"{len(anomalies)/len(data)*100:.1f}%")
        
        if not anomalies.empty:
            st.error("🚨 Recent Anomalies Detected")
            for _, anomaly in anomalies.tail(5).iterrows():
                st.warning(f"""
                    Anomaly at {anomaly['timestamp'].strftime('%Y-%m-%d %H:%M')}:
                    - Expected Load: {anomaly['baseline']:.2f}
                    - Actual Load: {anomaly['network_load']:.2f}
                    - Deviation: {anomaly['deviation']:.2f}
                    
                    Recommended Actions:
                    1. Check network segments for unusual activity
                    2. Compare with historical patterns
                    3. Monitor for pattern persistence
                """)

# Main app logic
if page == "Tower Placement":
    show_tower_placement()
elif page == "Network Monitor":
    show_network_monitor()
elif page == "Network Pattern Analysis":
    show_network_pattern_analysis()
elif page == "ROI Calculator":
    show_roi_calculator()

# Footer
st.markdown("---")
st.markdown("© 2024 BuildIQ by Ankita Avadhani - Powered by Machine Learning and Generative AI")
