import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Set aesthetics for plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load your data (adjust the filename as needed)
df = pd.read_parquet('data/joined_data.parquet')

# Get a quick overview
print(f"DataFrame dimensions: {df.shape}")
print("\nData types:")
print(df.dtypes.value_counts())

# Check for missing values
missing_data = df.isnull().sum()
missing_percent = 100 * missing_data / len(df)
missing_df = pd.DataFrame({'Missing Values': missing_data, 
                          'Percentage': missing_percent})
print("\nTop 10 columns with missing values:")
print(missing_df[missing_df['Missing Values'] > 0].sort_values('Missing Values', 
                                                              ascending=False).head(10))

# Look at basic statistics of the target variables
print("\nDropout rate statistics:")
print(df[['Taxa de Desistência Acumulada - TDA', 'Taxa de Desistência Anual - TADA']].describe())

# Distribution of dropout rates
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Taxa de Desistência Acumulada - TDA'].dropna(), kde=True)
plt.title('Distribution of Accumulated Dropout Rate')
plt.subplot(1, 2, 2)
sns.histplot(df['Taxa de Desistência Anual - TADA'].dropna(), kde=True)
plt.title('Distribution of Annual Dropout Rate')
plt.tight_layout()
plt.show()