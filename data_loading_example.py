#import fastparquet
import pandas as pd

# Load data from a Parquet file
df = pd.read_parquet('KMEM_data_train.parquet')

#display all columns
pd.set_option('display.max_columns', None)

# Display the DataFrame
print(df.head())