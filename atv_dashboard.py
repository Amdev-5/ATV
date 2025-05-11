import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from maintenance_optimizer import MaintenanceOptimizer, generate_sample_data
from spare_parts_forecast import SparePartsForecaster
import altair as alt

# Set page config
st.set_page_config(page_title="ATV Maintenance Dashboard", layout="wide")

# Title and description
st.title("🚗 ATV Maintenance Analytics Dashboard")
st.markdown("""
This dashboard provides comprehensive insights into ATV maintenance, defect patterns, 
and spare parts forecasting.
""")

# Load data
@st.cache_data
def load_defect_data():
    return pd.DataFrame({
        'Model': ['Ranger-1000', 'Ranger-1000', 'MRZR', 'MRZR', 'RZR'],
        'System': ['Eng sys', 'Txn sys', 'Fuel', 'Elect', 'Brake'],
        'Fault_Count': [1, 4, 2, 5, 2],
        'Detail_Defect': [
            'Head gasket w/o, Piston ring excessive wear',
            '02 x Half shaft w/o, 02 x CV joint shaft w/o',
            'Electronic fuel filter faulty, Spark plug short',
            '03 x Ignition switch faulty, 02 x Relay faulty',
            '04 x Calliper wear'
        ]
    })

@st.cache_data
def load_spares_data():
    return pd.DataFrame({
        'Part_Name': ['Fuel pump', 'Kit board', 'Air filter', 'Oil filter', 'Drive belt'],
        'Quantity_Consumed': [2, 8, 8, 14, 8],
        'Proposed_Stock': [2, 4, 8, 8, 6]
    })

# Load data
df_defects = load_defect_data()
df_spares = load_spares_data()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Defect Analysis", "Spare Parts Forecast", "Maintenance Optimization"])

with tab1:
    st.header("Defect Analysis")
    
    # Defect distribution by system and model
    st.subheader("Defect Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Bar plot using seaborn
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=df_defects, x='System', y='Fault_Count', hue='Model', ax=ax)
        plt.title('Defect Distribution by System and Model')
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
    with col2:
        # Heatmap using seaborn
        pivot_data = df_defects.pivot_table(index='Model', columns='System', 
                                          values='Fault_Count', fill_value=0)
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(pivot_data, annot=True, cmap='RdYlBu_r', ax=ax)
        plt.title('Defect Intensity Heatmap')
        st.pyplot(fig)
    
    # System-wise statistics
    st.subheader("System-wise Statistics")
    system_stats = df_defects.groupby('System').agg({
        'Fault_Count': ['sum', 'mean', 'max']
    }).round(2)
    system_stats.columns = ['Total Faults', 'Average Faults', 'Max Faults']
    st.dataframe(system_stats)

with tab2:
    st.header("Spare Parts Analysis and Forecast")
    
    # Initialize forecaster and generate data
    forecaster = SparePartsForecaster()
    historical_data = forecaster.generate_synthetic_data(df_spares)
    
    # Train the model first
    accuracy = forecaster.train(historical_data)
    st.info(f"Model R² Score: {accuracy:.2f}")
    
    # Current consumption analysis
    st.subheader("Current Spare Parts Consumption")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Current consumption vs proposed stock
        fig, ax = plt.subplots(figsize=(10, 6))
        x = range(len(df_spares['Part_Name']))
        width = 0.35
        ax.bar([i - width/2 for i in x], df_spares['Quantity_Consumed'], width, label='Consumed')
        ax.bar([i + width/2 for i in x], df_spares['Proposed_Stock'], width, label='Proposed')
        ax.set_xticks(x)
        ax.set_xticklabels(df_spares['Part_Name'], rotation=45)
        ax.legend()
        plt.title('Consumption vs Proposed Stock')
        st.pyplot(fig)
    
    with col2:
        # Stock ratio analysis using Altair
        chart = alt.Chart(df_spares).mark_circle().encode(
            x='Quantity_Consumed:Q',
            y='Proposed_Stock:Q',
            size='Quantity_Consumed:Q',
            color='Part_Name:N',
            tooltip=['Part_Name', 'Quantity_Consumed', 'Proposed_Stock']
        ).properties(
            width=400,
            height=300,
            title='Stock Analysis'
        )
        st.altair_chart(chart)
    
    # Forecast analysis
    st.subheader("Spare Parts Demand Forecast")
    
    # Select part for detailed analysis
    selected_part = st.selectbox("Select Part for Detailed Analysis", 
                               df_spares['Part_Name'].unique())
    
    # Generate forecast for selected part
    with st.spinner('Generating forecast...'):
        forecast = forecaster.predict_next_month(historical_data, selected_part)
        historical = historical_data[historical_data['Part_Name'] == selected_part]
        
        # Visualization
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(historical['Date'], historical['Demand'], label='Historical')
        ax.plot(forecast['Date'], forecast['Predicted_Demand'], '--', label='Forecast')
        ax.set_title(f'Demand Forecast for {selected_part}')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        
        # Forecast statistics
        stats = pd.DataFrame({
            'Metric': ['Average Daily Demand', 'Maximum Daily Demand', 
                      'Minimum Daily Demand', 'Recommended Stock Level'],
            'Value': [
                f"{forecast['Predicted_Demand'].mean():.1f}",
                str(forecast['Predicted_Demand'].max()),
                str(forecast['Predicted_Demand'].min()),
                str(int(forecast['Predicted_Demand'].mean() * 1.5))
            ]
        })
        st.dataframe(stats)

with tab3:
    st.header("Maintenance Optimization")
    
    # Generate and prepare data
    vehicle_data = generate_sample_data(10)
    optimizer = MaintenanceOptimizer()
    vehicle_data = optimizer.calculate_priority_score(vehicle_data)
    vehicle_data = optimizer.cluster_maintenance_needs(vehicle_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Cluster visualization
        fig, ax = plt.subplots(figsize=(10, 6))
        scatter = ax.scatter(vehicle_data['total_faults'], 
                           vehicle_data['days_since_last_maintenance'],
                           c=vehicle_data['maintenance_cluster'], 
                           s=vehicle_data['priority_score']*100)
        plt.colorbar(scatter)
        ax.set_xlabel('Total Faults')
        ax.set_ylabel('Days Since Last Maintenance')
        plt.title('Maintenance Clusters')
        st.pyplot(fig)
    
    with col2:
        # Priority score distribution
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(data=vehicle_data, x='model', y='priority_score', 
                   hue='maintenance_cluster', ax=ax)
        plt.title('Priority Score Distribution')
        plt.xticks(rotation=45)
        st.pyplot(fig)
    
    # Schedule visualization
    st.subheader("Maintenance Schedule")
    available_slots = []
    start_date = datetime.now()
    for i in range(5):
        for hour in [9, 14]:
            available_slots.append(
                (start_date + timedelta(days=i)).replace(hour=hour, minute=0)
            )
    
    schedule = optimizer.optimize_schedule(vehicle_data, available_slots, mechanics_per_slot=2)
    st.dataframe(schedule)

# Footer
st.markdown("---")
st.markdown("Dashboard created for ATV Maintenance Analysis and Prediction System") 