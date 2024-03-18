import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Load the data
data_path = 'data.xlsx'  # Change this to your actual file path
data = pd.read_excel(data_path)

# Convert columns to string to handle datetime format conversion
data.columns = data.columns.map(str)

# Melt the dataframe to long format
melted_data = pd.melt(data, id_vars=['name'], var_name='datetime', value_name='people_count')
melted_data['datetime'] = pd.to_datetime(melted_data['datetime'], errors='coerce')

# Filter out the training and validation sets
train_data = melted_data[melted_data['datetime'].dt.date < datetime(2023, 12, 29).date()]
validation_data = melted_data[melted_data['datetime'].dt.date == datetime(2023, 12, 29).date()]

# Add day of week and hour as features
for df in [train_data, validation_data]:
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour

# Prepare training and validation sets
X_train = train_data[['day_of_week', 'hour']]
y_train = train_data['people_count']
X_validation = validation_data[['day_of_week', 'hour']]
y_validation = validation_data['people_count']

# Train a Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions on the validation set
validation_predictions = rf_model.predict(X_validation)
validation_data['predictions'] = validation_predictions

# Calculate MSE and MAPE
mse = mean_squared_error(y_validation, validation_predictions)
mape = mean_absolute_percentage_error(y_validation, validation_predictions)

# Visualization for a sample facility
sample_facility_name = train_data['name'].unique()[2]
sample_train_data = train_data[train_data['name'] == sample_facility_name]

sample_validation_data = validation_data[validation_data['name'] == sample_facility_name]  # Change 'Sample Facility' to a real facility name from your dataset
plt.figure(figsize=(12, 6))
plt.plot(sample_train_data['datetime'], sample_train_data['people_count'], label=f'{sample_facility_name} Training', color='green')

# Plot validation actual data
plt.plot(sample_validation_data['datetime'], sample_validation_data['people_count'], label=f'{sample_facility_name} Actual', marker='o', linestyle='-', color='blue')

# Plot validation predicted data
plt.plot(sample_validation_data['datetime'], sample_validation_data['predictions'], label=f'{sample_facility_name} Predicted', linestyle='--', marker='x', color='red')

# Setting the title and labels
plt.title(f'People Counts for {sample_facility_name}: Random Forest')
plt.xlabel('Time')
plt.ylabel('People Count')
plt.legend()
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.tight_layout()  # Adjust layout to not cut off labels
plt.show()