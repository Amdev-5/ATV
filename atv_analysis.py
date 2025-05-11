import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
import warnings
warnings.filterwarnings('ignore')

# Data Structure for Defect Analysis
defect_data = {
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
}

# Create DataFrame
df_defects = pd.DataFrame(defect_data)

# Spares consumption data
spares_data = {
    'Part_Name': ['Fuel pump', 'Kit board', 'Air filter', 'Oil filter', 'Drive belt'],
    'Quantity_Consumed': [2, 8, 8, 14, 8],
    'Proposed_Stock': [2, 4, 8, 8, 6]
}
df_spares = pd.DataFrame(spares_data)

def analyze_defect_patterns():
    """Analyze and visualize defect patterns across different ATV models"""
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_defects, x='System', y='Fault_Count', hue='Model')
    plt.title('Defect Distribution by System and Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def analyze_spares_consumption():
    """Analyze spares consumption patterns"""
    plt.figure(figsize=(10, 6))
    plt.bar(df_spares['Part_Name'], df_spares['Quantity_Consumed'])
    plt.title('Spares Consumption Analysis')
    plt.xticks(rotation=45)
    plt.ylabel('Quantity Consumed')
    plt.tight_layout()
    plt.show()

class ATVDefectPredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.label_encoder = LabelEncoder()
    
    def prepare_data(self, df):
        # Encode categorical variables
        df['Model_Encoded'] = self.label_encoder.fit_transform(df['Model'])
        df['System_Encoded'] = self.label_encoder.fit_transform(df['System'])
        
        # Features and target
        X = df[['Model_Encoded', 'System_Encoded']]
        y = df['Fault_Count']
        
        return train_test_split(X, y, test_size=0.2, random_state=42)
    
    def train(self, df):
        X_train, X_test, y_train, y_test = self.prepare_data(df)
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)
    
    def predict(self, model, system):
        # Encode input
        model_encoded = self.label_encoder.transform([model])
        system_encoded = self.label_encoder.transform([system])
        
        return self.model.predict([[model_encoded[0], system_encoded[0]]])[0]

def main():
    # Analyze current patterns
    analyze_defect_patterns()
    analyze_spares_consumption()
    
    # Train predictive model
    predictor = ATVDefectPredictor()
    accuracy = predictor.train(df_defects)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Example prediction
    predicted_faults = predictor.predict('Ranger-1000', 'Eng sys')
    print(f"Predicted number of faults for Ranger-1000 Engine System: {predicted_faults}")

if __name__ == "__main__":
    main() 