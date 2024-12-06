
import pandas as pd

# Load the CSV file
file_path = 'data/run_table.csv'
data = pd.read_csv(file_path)

# Extract run and repetition numbers
data[['run', 'repetition']] = data['__run_id'].str.extract(r'run_(\d+)_repetition_(\d+)').astype(int)

# Sort the data by run and repetition
sorted_data = data.sort_values(by=['run', 'repetition']).reset_index(drop=True)

# Save the sorted data to a new CSV file
sorted_data.to_csv('sorted_run_table.csv', index=False)
