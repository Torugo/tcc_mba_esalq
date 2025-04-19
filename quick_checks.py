import pandas as pd

# Load just a sample to check structure
df_sample = pd.read_parquet("joined_data.parquet", engine='pyarrow')

# General info
print(f"DataFrame shape: {df_sample.shape}")
print(f"Memory usage: {df_sample.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")

# Check data types
print("\nData types:")
print(df_sample.dtypes)

# Check target variable
print("\nTarget variable info:")
print(df_sample["Taxa de Conclusão Acumulada - TCA"].describe())
print(f"Missing values: {df_sample['Taxa de Conclusão Acumulada - TCA'].isnull().sum()}")

# Check cardinality of key categorical columns
print("\nUnique values in categorical columns:")
for col in ["Categoria Administrativa", "Organização Acadêmica", 
            "Grau Acadêmico", "Nome da Grande Área do Curso segundo a classificação CINE BRASIL"]:
    if col in df_sample.columns:
        print(f"{col}: {df_sample[col].nunique()} unique values")