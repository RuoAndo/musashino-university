import csv
import numpy as np
import sys
import os
import re
import matplotlib.pyplot as plt
from datetime import datetime

def extract_date_from_filename(file_name):
    # Extract date from filename (format: YYYYMMDD)
    match = re.search(r'\d{8}', file_name)  # Match 8-digit date
    if match:
        # Convert YYYYMMDD to YYYY-MM-DD
        raw_date = match.group(0)
        formatted_date = f"{raw_date[:4]}-{raw_date[4:6]}-{raw_date[6:]}"
        return formatted_date
    return None

def calculate_statistics(file_path, column_index):
    scores = []
    
    # Read the file
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        for row in reader:
            try:
                # Extract numerical data (7th column is index 6)
                score = float(row[column_index])
                scores.append(score)
            except (ValueError, IndexError):
                # Skip rows that cannot be converted to float
                continue

    # Calculate mean, variance, and max
    if scores:
        mean_score = np.mean(scores)
        variance_score = np.var(scores)
        max_score = np.max(scores)
        return mean_score, variance_score, max_score
    else:
        return None, None, None

def process_directory(directory_path, column_index):
    data_points = []  # Store date and statistics

    for file_name in os.listdir(directory_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(directory_path, file_name)
            date_str = extract_date_from_filename(file_name)  # Extract date from filename
            
            if date_str:
                mean, variance, max_value = calculate_statistics(file_path, column_index)
                if mean is not None:
                    data_points.append((date_str, mean, variance, max_value))
                    print(f"Date: {date_str}, Mean: {mean}, Variance: {variance}, Max: {max_value}")
                else:
                    print(f"File: {file_name} - No valid scores found.")
    
    # Sort data by date
    data_points.sort(key=lambda x: datetime.strptime(x[0], '%Y-%m-%d'))

    # Prepare data for plotting
    dates = [datetime.strptime(dp[0], '%Y-%m-%d') for dp in data_points]
    means = [dp[1] for dp in data_points]
    variances = [dp[2] for dp in data_points]
    max_values = [dp[3] for dp in data_points]

    # Plot time series (subplots)
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # Plot mean values
    axes[0].plot(dates, means, marker='o', linestyle='-', label='Mean', color='blue')
    axes[0].set_ylabel('Mean')
    axes[0].grid(True)
    axes[0].legend()

    # Plot variances
    axes[1].plot(dates, variances, marker='s', linestyle='-', label='Variance', color='orange')
    axes[1].set_ylabel('Variance')
    axes[1].grid(True)
    axes[1].legend()

    # Plot maximum values
    axes[2].plot(dates, max_values, marker='^', linestyle='-', label='Max', color='green')
    axes[2].set_xlabel('Date')
    axes[2].set_ylabel('Max')
    axes[2].grid(True)
    axes[2].legend()

    # Add title
    fig.suptitle('Time Series of Mean, Variance, and Max', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    # Get directory path from command-line arguments
    if len(sys.argv) != 2:
        print("Usage: python script_name.py <directory_path>")
        sys.exit(1)

    directory_path = sys.argv[1]  # Directory path
    column_index = 6              # 7th column is index 6
    process_directory(directory_path, column_index)
