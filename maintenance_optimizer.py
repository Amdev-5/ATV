import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import linear_sum_assignment

class MaintenanceOptimizer:
    def __init__(self):
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        
    def calculate_priority_score(self, vehicle_data):
        """Calculate priority scores based on various factors"""
        # Normalize factors
        max_faults = vehicle_data['total_faults'].max()
        max_age = vehicle_data['days_since_last_maintenance'].max()
        
        # Calculate weighted score
        vehicle_data['priority_score'] = (
            0.4 * (vehicle_data['total_faults'] / max_faults) +
            0.3 * (vehicle_data['days_since_last_maintenance'] / max_age) +
            0.3 * vehicle_data['critical_system_faults']
        )
        
        return vehicle_data
    
    def cluster_maintenance_needs(self, vehicle_data):
        """Cluster vehicles based on maintenance needs"""
        features = vehicle_data[['total_faults', 'days_since_last_maintenance', 
                               'critical_system_faults', 'priority_score']].values
        
        # Normalize features
        features = (features - features.mean(axis=0)) / features.std(axis=0)
        
        # Perform clustering
        clusters = self.kmeans.fit_predict(features)
        vehicle_data['maintenance_cluster'] = clusters
        
        return vehicle_data
    
    def optimize_schedule(self, vehicle_data, available_slots, mechanics_per_slot):
        """Optimize maintenance schedule using Hungarian algorithm"""
        n_vehicles = len(vehicle_data)
        n_slots = len(available_slots)
        
        # Create cost matrix based on priority scores and time slots
        cost_matrix = np.zeros((n_vehicles, n_slots))
        
        for i in range(n_vehicles):
            for j in range(n_slots):
                # Higher priority score means lower cost (we want to schedule high priority first)
                cost_matrix[i, j] = 1 - vehicle_data.iloc[i]['priority_score']
                
                # Add penalty for later time slots for high priority vehicles
                cost_matrix[i, j] += j * 0.1 * vehicle_data.iloc[i]['priority_score']
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Create schedule
        schedule = []
        for i, j in zip(row_ind, col_ind):
            if j < n_slots:  # Only include valid assignments
                schedule.append({
                    'vehicle': vehicle_data.iloc[i]['vehicle_id'],
                    'time_slot': available_slots[j],
                    'priority_score': vehicle_data.iloc[i]['priority_score'],
                    'maintenance_cluster': vehicle_data.iloc[i]['maintenance_cluster']
                })
        
        return pd.DataFrame(schedule)

def generate_sample_data(n_vehicles=10):
    """Generate sample vehicle maintenance data"""
    np.random.seed(42)
    
    data = []
    for i in range(n_vehicles):
        data.append({
            'vehicle_id': f'ATV_{i+1:02d}',
            'total_faults': np.random.randint(0, 10),
            'days_since_last_maintenance': np.random.randint(0, 100),
            'critical_system_faults': np.random.randint(0, 3),
            'model': np.random.choice(['Ranger-1000', 'MRZR', 'RZR'])
        })
    
    return pd.DataFrame(data)

def visualize_maintenance_clusters(vehicle_data):
    """Visualize maintenance clusters"""
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=vehicle_data, x='total_faults', y='days_since_last_maintenance',
                    hue='maintenance_cluster', size='priority_score',
                    palette='deep', sizes=(50, 200))
    plt.title('Maintenance Clusters')
    plt.xlabel('Total Faults')
    plt.ylabel('Days Since Last Maintenance')
    plt.show()

def visualize_schedule(schedule):
    """Visualize maintenance schedule"""
    plt.figure(figsize=(12, 6))
    
    # Create timeline plot
    for i, row in schedule.iterrows():
        plt.scatter(pd.to_datetime(row['time_slot']), i, 
                   s=100 * row['priority_score'] + 50,
                   c=[plt.cm.Set3(row['maintenance_cluster'])],
                   label=f'Cluster {row["maintenance_cluster"]}' if i == 0 else "")
        
        plt.text(pd.to_datetime(row['time_slot']), i, 
                row['vehicle'], fontsize=8,
                horizontalalignment='right')
    
    plt.title('Maintenance Schedule')
    plt.xlabel('Time Slot')
    plt.ylabel('Schedule Order')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

def main():
    # Generate sample data
    vehicle_data = generate_sample_data(10)
    
    # Create optimizer
    optimizer = MaintenanceOptimizer()
    
    # Calculate priority scores
    vehicle_data = optimizer.calculate_priority_score(vehicle_data)
    
    # Cluster vehicles
    vehicle_data = optimizer.cluster_maintenance_needs(vehicle_data)
    
    # Generate available time slots (next 5 days, 2 slots per day)
    available_slots = []
    start_date = datetime.now()
    for i in range(5):
        for hour in [9, 14]:  # Morning and afternoon slots
            available_slots.append(
                (start_date + timedelta(days=i)).replace(hour=hour, minute=0)
            )
    
    # Optimize schedule
    schedule = optimizer.optimize_schedule(vehicle_data, available_slots, mechanics_per_slot=2)
    
    # Visualize results
    visualize_maintenance_clusters(vehicle_data)
    visualize_schedule(schedule)
    
    # Print schedule
    print("\nOptimized Maintenance Schedule:")
    for _, row in schedule.iterrows():
        print(f"Vehicle {row['vehicle']} (Priority: {row['priority_score']:.2f}) "
              f"scheduled for {row['time_slot']}")

if __name__ == "__main__":
    main() 