import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Read the CSV file
df = pd.read_csv('sensor_data_nosein_noseout_10_with_anomalies.csv')

# Convert Timestamp to datetime
df['Timestamp'] = pd.to_datetime(df['Timestamp'])

# Calculate the floor of each timestamp to the nearest second
df['second'] = df['Timestamp'].dt.floor('s')

# Count samples per second
samples_per_second = df.groupby('second').size()

# Calculate statistics
average_samples_per_second = samples_per_second.mean()
min_samples = samples_per_second.min()
max_samples = samples_per_second.max()
std_samples = samples_per_second.std()

# Print statistics
print("\nSample Rate Statistics:")
print(f"Average samples per second: {average_samples_per_second:.2f}")
print(f"Minimum samples per second: {min_samples}")
print(f"Maximum samples per second: {max_samples}")
print(f"Standard deviation: {std_samples:.2f}")

# Plot the distribution
plt.figure(figsize=(12, 6))
counts, bins, patches = plt.hist(samples_per_second, bins=50, edgecolor='black')

# Add count labels on top of each bar
for i in range(len(counts)):
    if counts[i] >= 1:  # Only show labels for bins with at least 1 count
        count = int(counts[i])  # Convert to integer
        plt.text(bins[i] + (bins[i+1]-bins[i])/2, counts[i], 
                str(count), 
                ha='center', va='bottom')

plt.title('Distribution of Samples per Second')
plt.xlabel('Number of Samples')
plt.ylabel('Frequency')
plt.grid(True, alpha=0.3)

# Add vertical line for average
plt.axvline(average_samples_per_second, color='red', linestyle='--', 
            label=f'Average: {average_samples_per_second:.2f} samples/sec')
plt.legend()

# Save the plot
plt.savefig('sample_rate_distribution.png')
plt.close()

# Print the distribution table
print("\nSample Rate Distribution Table:")
print(samples_per_second.describe())

# Print first few seconds as example
print("\nExample of first few seconds:")
first_few_seconds = samples_per_second.head()
print(first_few_seconds)
