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

print(train_data)

# Function to calculate historical average people count for each facility and hour
def calculate_historical_averages(dataset):
    # We group by facility name and the hour of the day, then calculate the mean people count
    dataset['hour'] = dataset['datetime'].dt.hour
    averages = dataset.groupby(['name', 'hour']).mean().reset_index()
    return averages

# Apply the function to the training data
historical_averages = calculate_historical_averages(train_data)

# Function to make predictions based on the historical averages for a new dataset
def predict_using_averages(dataset, averages):
    # Merge the predictions with the new dataset based on facility name and hour
    dataset['hour'] = dataset['datetime'].dt.hour
    predictions = pd.merge(dataset, averages, on=['name', 'hour'], how='left')
    # We'll return the dataset with a new column for the predicted people count
    return predictions

# Use the historical averages to predict the people count for the validation dataset
validation_predictions = predict_using_averages(validation_data, historical_averages)
validation_predictions.rename(columns={'people_count_x': 'actual', 'people_count_y': 'predicted'}, inplace=True)

# Evaluate the model using MSE and MAPE
mse = mean_squared_error(validation_predictions['actual'], validation_predictions['predicted'])
mape = mean_absolute_percentage_error(validation_predictions['actual'], validation_predictions['predicted'])

# Visualization: Plot actual vs predicted counts for the first 10 entries as an example
# Visualization: Plot training, actual vs predicted counts for a sample facility
plt.figure(figsize=(12, 6))

# Select a sample facility for visualization
sample_facility_name = train_data['name'].unique()[2]  # You can change this to any facility name from your dataset

# Filter data for the sample facility
sample_train_data = train_data[train_data['name'] == sample_facility_name]
sample_validation_data = validation_predictions[validation_predictions['name'] == sample_facility_name]

# Plot training data
plt.plot(sample_train_data['datetime'], sample_train_data['people_count'], label=f'{sample_facility_name} Training', color='green')

# Plot validation actual data
plt.plot(sample_validation_data['datetime'], sample_validation_data['actual'], label=f'{sample_facility_name} Actual', marker='o', linestyle='-', color='blue')

# Plot validation predicted data
plt.plot(sample_validation_data['datetime'], sample_validation_data['predicted'], label=f'{sample_facility_name} Predicted', linestyle='--', marker='x', color='red')

# Setting the title and labels
plt.title(f'People Counts for {sample_facility_name}: Based on historical averages')
plt.xlabel('Time')
plt.ylabel('People Count')
plt.legend()
plt.xticks(rotation=45)  # Rotate dates for better readability
plt.tight_layout()  # Adjust layout to not cut off labels
plt.show()