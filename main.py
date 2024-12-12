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
    ["Tower Placement", "Network Monitor", "Churn Risk", "ROI Calculator"])

[Previous data generation functions remain the same...]

def show_tower_placement():
    st.title("🗼 Tower Infrastructure Management")
    
    tab1, tab2, tab3 = st.tabs(["📍 Location Analysis", "🔧 Maintenance", "🌤️ Weather Impact"])
    
    with tab1:
        st.subheader("Smart Tower Placement Map")
        st.write("AI-powered analysis for optimal tower locations")
        
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

    with tab2:
        st.subheader("🔧 AI-Driven Predictive Maintenance")
        maintenance_data = generate_maintenance_data()
        
        # Show maintenance schedule
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

    with tab3:
        st.subheader("🌤️ Real-Time Weather Impact Analysis")
        
        # Generate weather impact data
        weather_data = pd.DataFrame({
            'condition': ['Clear', 'Rain', 'Storm', 'Snow', 'Fog'],
            'signal_strength': [95, 80, 60, 70, 75],
            'affected_towers': np.random.randint(0, 20, 5)
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Weather impact visualization
            fig = px.bar(weather_data, x='condition', y='signal_strength',
                         title='Signal Strength by Weather Condition',
                         labels={'condition': 'Weather', 'signal_strength': 'Signal Strength (%)'})
            st.plotly_chart(fig)
        
        with col2:
            # Real-time weather alerts
            st.subheader("Active Weather Alerts")
            alerts = [
                {"severity": "High", "condition": "Storm", "location": "North Sector"},
                {"severity": "Medium", "condition": "Rain", "location": "West Sector"},
            ]
            
            for alert in alerts:
                color = "red" if alert["severity"] == "High" else "orange"
                st.markdown(f":{color}[{alert['severity']}]: {alert['condition']} in {alert['location']}")
            
            # Impact statistics
            st.metric("Affected Towers", f"{weather_data['affected_towers'].sum()}")
            st.metric("Average Signal Impact", 
                     f"{(100 - weather_data['signal_strength'].mean()):.1f}% degradation")

[Previous show_network_monitor(), show_churn_risk(), and show_roi_calculator() functions remain the same...]

# Main app logic
if page == "Tower Placement":
    show_tower_placement()
elif page == "Network Monitor":
    show_network_monitor()
elif page == "Churn Risk":
    show_churn_risk()
elif page == "ROI Calculator":
    show_roi_calculator()

# Footer
st.markdown("---")
st.markdown("© 2024 BuildIQ by Ankita Avadhani - Powered by Machine Learning and Generative AI")
