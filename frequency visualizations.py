import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the training dataset
train_data = pd.read_excel('train.xlsx')

# Set the style for the plots
sns.set(style="whitegrid")

# Handle the temperature unit issue - convert Kelvin to Celsius
print("Converting temperature units...")
temp_mask = train_data['Avg_Temperature'] > 100  # Threshold to identify Kelvin values
train_data.loc[temp_mask, 'Avg_Temperature'] = train_data.loc[temp_mask, 'Avg_Temperature'] - 273.15

# Visualize the distribution of Average Temperature after conversion
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Avg_Temperature'], kde=True, color='blue')
plt.title('Distribution of Average Temperature (Converted to Celsius)')
plt.xlabel('Temperature (°C)')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of Radiation (W/m²)
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Radiation'], kde=True, color='red')
plt.title('Distribution of Radiation (W/m²)')
plt.xlabel('Radiation (W/m²)')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of Rain Amount (mm)
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Rain_Amount'], kde=True, color='green')
plt.title('Distribution of Rain Amount (mm)')
plt.xlabel('Rain Amount (mm)')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of Wind Speed (km/h)
plt.figure(figsize=(10, 6))
sns.histplot(train_data['Wind_Speed'], kde=True, color='orange')
plt.title('Distribution of Wind Speed (km/h)')
plt.xlabel('Wind Speed (km/h)')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of Wind Direction (°) 
plt.figure(figsize=(12, 6))
sns.countplot(data=train_data, x='Wind_Direction', palette='viridis')
plt.title('Frequency Distribution of Wind Direction (°)')
plt.xlabel('Wind Direction (°)')
plt.ylabel('Count')
plt.xticks(ticks=range(0, 360, 45), labels=[str(i) for i in range(0, 360, 45)])
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
