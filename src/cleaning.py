import pandas as pd
import glob

# Path to the folder containing Excel files
folder_path = "./dataset"  # Adjust this path to the location of your 'dataset' folder

# Get all Excel files in the folder
excel_files = glob.glob(f"{folder_path}/*.xlsx")

# List to store DataFrames
dataframes = []

# Loop through each Excel file
for file in excel_files:
    # Read all sheets from the current Excel file
    sheets = pd.read_excel(file, sheet_name=None)
    for sheet_name, df in sheets.items():
        # Add a column to identify the source file and sheet (optional)
        df['Source_File'] = file
        df['Source_Sheet'] = sheet_name
        dataframes.append(df)

# Concatenate all DataFrames
concatenated_df = pd.concat(dataframes, ignore_index=True)

# Save the concatenated data to a new Excel file
output_file = "combined_dataset.xlsx"
concatenated_df.to_excel(output_file, index=False)

print(f"All sheets from the folder '{folder_path}' have been concatenated and saved to {output_file}.")

import pandas as pd

# Load the dataset
file_path = "combined_dataset.xlsx"  # Replace with your file path
data = pd.read_excel(file_path)

# Step 1: Check for missing values
print("Missing Values Before Cleaning:")
print(data.isnull().sum())

# Drop columns with more than 50% missing values
threshold = 0.5  # Adjust threshold as needed
data = data.loc[:, data.isnull().mean() < threshold]

# Fill remaining missing values (median for numerical, mode for categorical)
for column in data.columns:
    if data[column].dtype in ["float64", "int64"]:
        data[column].fillna(data[column].median(), inplace=True)
    else:
        data[column].fillna(data[column].mode()[0], inplace=True)

# Step 2: Remove duplicate rows
data = data.drop_duplicates()
print("\nData Shape After Removing Duplicates:", data.shape)

# Step 3: Handle outliers (using IQR method)
numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns

for column in numerical_columns:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    # Filter out outliers
    data = data[(data[column] >= lower_bound) & (data[column] <= upper_bound)]

print("\nData Shape After Outlier Removal:", data.shape)

# Step 4: Standardize column names
data.columns = [col.strip().lower().replace(" ", "_") for col in data.columns]
print("\nColumn Names After Standardization:")
print(data.columns)

# Step 5: Convert all string columns to lowercase
string_columns = data.select_dtypes(include=["object"]).columns

for column in string_columns:
    data[column] = data[column].str.lower()

print("\nSample Data After Converting Strings to Lowercase:")
print(data.head())

# Step 6: Convert data types (example: date conversion, if applicable)
# Uncomment and adjust if you have date columns
# data['date_column'] = pd.to_datetime(data['date_column'], errors='coerce')

# Save the cleaned dataset
cleaned_file = "cleaned_combined_dataset_lowercase.xlsx"
data.to_excel(cleaned_file, index=False)

print(f"\nData cleaning complete. Cleaned dataset saved as {cleaned_file}.")