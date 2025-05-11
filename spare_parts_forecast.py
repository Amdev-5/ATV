import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class SparePartsForecaster:
    def __init__(self):
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def generate_synthetic_data(self, parts_data):
        """Generate synthetic time series data for demonstration"""
        dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
        parts = parts_data['Part_Name'].unique()
        
        data = []
        for part in parts:
            base_demand = parts_data[parts_data['Part_Name'] == part]['Quantity_Consumed'].values[0]
            for date in dates:
                # Add seasonality and trend
                seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.dayofyear / 365)
                trend = 0.001 * (date - dates[0]).days
                
                demand = base_demand * seasonal_factor * (1 + trend)
                demand = max(0, int(demand + np.random.normal(0, 0.5)))
                
                data.append({
                    'Date': date,
                    'Part_Name': part,
                    'Demand': demand
                })
        
        return pd.DataFrame(data)
    
    def prepare_features(self, df):
        """Prepare features for the model"""
        df = df.copy()
        
        # Time-based features
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['Month'] = df['Date'].dt.month
        df['DayOfMonth'] = df['Date'].dt.day
        
        # Lag features (previous 7 days demand)
        for i in range(1, 8):
            df[f'Demand_Lag_{i}'] = df.groupby('Part_Name')['Demand'].shift(i)
        
        # Rolling mean features
        df['Rolling_Mean_7'] = df.groupby('Part_Name')['Demand'].rolling(7).mean().reset_index(0, drop=True)
        df['Rolling_Mean_30'] = df.groupby('Part_Name')['Demand'].rolling(30).mean().reset_index(0, drop=True)
        
        # Drop NaN values from lag features
        df = df.dropna()
        
        return df
    
    def train(self, df):
        """Train the forecasting model"""
        prepared_data = self.prepare_features(df)
        
        # Prepare features and target
        feature_cols = ['DayOfWeek', 'Month', 'DayOfMonth', 
                       'Demand_Lag_1', 'Demand_Lag_2', 'Demand_Lag_3',
                       'Demand_Lag_4', 'Demand_Lag_5', 'Demand_Lag_6',
                       'Demand_Lag_7', 'Rolling_Mean_7', 'Rolling_Mean_30']
        
        X = prepared_data[feature_cols]
        y = prepared_data['Demand']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        
        return self.model.score(X_scaled, y)
    
    def predict_next_month(self, df, part_name):
        """Predict demand for next month for a specific part"""
        if not self.is_fitted:
            # Train the model if not already trained
            self.train(df)
        
        last_date = df['Date'].max()
        future_dates = pd.date_range(start=last_date + timedelta(days=1),
                                   end=last_date + timedelta(days=30),
                                   freq='D')
        
        future_data = []
        current_data = df[df['Part_Name'] == part_name].copy()
        
        for future_date in future_dates:
            new_row = pd.DataFrame({
                'Date': [future_date],
                'Part_Name': [part_name],
                'Demand': [0]  # Placeholder
            })
            current_data = pd.concat([current_data, new_row])
            prepared_row = self.prepare_features(current_data).iloc[-1:]
            
            feature_cols = ['DayOfWeek', 'Month', 'DayOfMonth',
                          'Demand_Lag_1', 'Demand_Lag_2', 'Demand_Lag_3',
                          'Demand_Lag_4', 'Demand_Lag_5', 'Demand_Lag_6',
                          'Demand_Lag_7', 'Rolling_Mean_7', 'Rolling_Mean_30']
            
            X = prepared_row[feature_cols]
            X_scaled = self.scaler.transform(X)
            prediction = self.model.predict(X_scaled)[0]
            
            future_data.append({
                'Date': future_date,
                'Predicted_Demand': max(0, int(prediction))
            })
            
            current_data.iloc[-1, current_data.columns.get_loc('Demand')] = prediction
            
        return pd.DataFrame(future_data)

def visualize_forecast(historical_data, forecast_data, part_name):
    """Visualize historical demand and forecast"""
    plt.figure(figsize=(12, 6))
    
    # Plot historical data
    historical = historical_data[historical_data['Part_Name'] == part_name]
    plt.plot(historical['Date'], historical['Demand'], 
             label='Historical Demand', color='blue')
    
    # Plot forecast
    plt.plot(forecast_data['Date'], forecast_data['Predicted_Demand'],
             label='Forecast', color='red', linestyle='--')
    
    plt.title(f'Demand Forecast for {part_name}')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    # Load initial spare parts data
    spares_data = pd.DataFrame({
        'Part_Name': ['Fuel pump', 'Kit board', 'Air filter', 'Oil filter', 'Drive belt'],
        'Quantity_Consumed': [2, 8, 8, 14, 8],
        'Proposed_Stock': [2, 4, 8, 8, 6]
    })
    
    # Initialize forecaster
    forecaster = SparePartsForecaster()
    
    # Generate synthetic historical data
    historical_data = forecaster.generate_synthetic_data(spares_data)
    
    # Train model
    accuracy = forecaster.train(historical_data)
    print(f"Model R² Score: {accuracy:.2f}")
    
    # Generate and visualize forecast for each part
    for part in spares_data['Part_Name']:
        forecast = forecaster.predict_next_month(historical_data, part)
        visualize_forecast(historical_data, forecast, part)
        
        # Print summary statistics
        print(f"\nForecast Summary for {part}:")
        print(f"Average Daily Demand: {forecast['Predicted_Demand'].mean():.1f}")
        print(f"Maximum Daily Demand: {forecast['Predicted_Demand'].max()}")
        print(f"Minimum Daily Demand: {forecast['Predicted_Demand'].min()}")
        print(f"Recommended Stock Level: {int(forecast['Predicted_Demand'].mean() * 1.5)}")

if __name__ == "__main__":
    main() 