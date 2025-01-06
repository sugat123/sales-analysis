import pandas as pd
import numpy as np

# Function to clean numerical columns
def clean_numerical_columns(df, columns):
    for col in columns:
        # Replace NaN with the median
        df[col] = df[col].fillna(df[col].median())

        # Handle outliers using the IQR method
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Replace outliers with NaN and fill with the median
        df[col] = np.where((df[col] < lower_bound) | (df[col] > upper_bound), np.nan, df[col])
        df[col] = df[col].fillna(df[col].median())
    return df

# Function to clean categorical columns
def clean_categorical_columns(df, columns):
    for col in columns:
        # Replace NaN with the mode
        df[col] = df[col].fillna(df[col].mode()[0])
        # Convert to lowercase
        df[col] = df[col].str.lower()
    return df

# Function to clean the saledate column
def clean_saledate_column(df, column):
    # Remove timezone information in parentheses
    df[column] = df[column].str.replace(r'\s*\(.*\)', '', regex=True)
    # Convert to datetime
    df[column] = pd.to_datetime(df[column], format='%a %b %d %Y %H:%M:%S GMT%z', errors='coerce', utc=True)
    return df

# Function to clean the entire dataset
def clean_vehicle_sales_data(filepath, save_filepath=None):
    # Load the dataset
    df = pd.read_csv(filepath)

    # Clean numerical columns
    numerical_columns = ['condition', 'odometer', 'mmr', 'sellingprice']
    df = clean_numerical_columns(df, numerical_columns)

    # Clean categorical columns
    categorical_columns = ['make', 'model', 'trim', 'body', 'transmission', 'color', 'interior', 'seller', 'state']
    df = clean_categorical_columns(df, categorical_columns)

    # Drop rows where 'vin' is null
    df = df.dropna(subset=['vin'])

    # Clean the 'saledate' column
    df = clean_saledate_column(df, 'saledate')
    df = df.dropna(subset=['saledate'])

    # Convert data types
    df['year'] = df['year'].astype(int)
    for col in numerical_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Remove unrealistic selling prices
    df = df[df['sellingprice'] > 1]

    # Drop duplicate values
    df.drop_duplicates(inplace=True)

    # Add new features: year and month of sale
    df['sale_year'] = df['saledate'].dt.year
    df['sale_month'] = df['saledate'].dt.month

    # Save cleaned data (optional)
    if save_filepath:
        df.to_csv(save_filepath, index=False)

    return df

# Run the cleaning process
cleaned_data = clean_vehicle_sales_data(
    filepath="car_prices.csv", 
    save_filepath="cleaned_vehicle_sales_data.csv"
)

# Verify the cleaned DataFrame
print(cleaned_data.describe())
