import pandas as pd
df = pd.read_parquet("data/joined_data.parquet")
print(df.columns.to_list())

# Show DataFrame dimensions (rows, columns)
print(f"DataFrame dimensions: {df.shape}")

# For more details, you can also do:
print(f"Number of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")