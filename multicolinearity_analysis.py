import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import re
import warnings
warnings.filterwarnings('ignore')

def identify_dictionary_based_exclusions():
    """
    Create a list of column patterns to exclude based on the data dictionary
    
    Returns:
    -------
    exclude_patterns : list
        List of dictionaries with regex patterns and reasons for exclusion
    """
    # Define patterns for columns that can be excluded based on the dictionary
    exclude_patterns = [
        # Identification codes - usually categorical identifiers
        {"pattern": re.compile(r'^CO_'), "reason": "Code/ID column - categorical identifier"},
        {"pattern": re.compile(r'^NU_'), "reason": "Numeric ID - likely unique identifier"},
        
        # Descriptive text fields with no numerical relevance
        {"pattern": re.compile(r'^NO_'), "reason": "Name field - textual data"},
        {"pattern": re.compile(r'^SG_'), "reason": "Abbreviation field - textual data"},
        {"pattern": re.compile(r'^DS_'), "reason": "Description field - textual data"},
        
        # Redundant breakdowns by gender - keep only the totals
        {"pattern": re.compile(r'^QT_.*_FEM$'), "reason": "Gender breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_MASC$'), "reason": "Gender breakdown - redundant with total"},
        
        # Redundant breakdowns by time of day - keep only the totals
        {"pattern": re.compile(r'^QT_.*_DIURNO$'), "reason": "Time breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_NOTURNO$'), "reason": "Time breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_EAD$'), "reason": "Mode breakdown - redundant with total"},
        
        # Redundant breakdowns by age - keep only the totals
        {"pattern": re.compile(r'^QT_.*_0_17$'), "reason": "Age breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_18_24$'), "reason": "Age breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_25_29$'), "reason": "Age breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_30_34$'), "reason": "Age breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_35_39$'), "reason": "Age breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_40_49$'), "reason": "Age breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_50_59$'), "reason": "Age breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_60_MAIS$'), "reason": "Age breakdown - redundant with total"},
        
        # Redundant breakdowns by race - keep only the totals
        {"pattern": re.compile(r'^QT_.*_BRANCA$'), "reason": "Race breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_PRETA$'), "reason": "Race breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_PARDA$'), "reason": "Race breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_AMARELA$'), "reason": "Race breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_INDIGENA$'), "reason": "Race breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_CORND$'), "reason": "Race breakdown - redundant with total"},
        
        # Redundant nationality breakdowns - keep only the totals
        {"pattern": re.compile(r'^QT_.*_NACBRAS$'), "reason": "Nationality breakdown - redundant with total"},
        {"pattern": re.compile(r'^QT_.*_NACESTRANG$'), "reason": "Nationality breakdown - redundant with total"},
        
        # Type indicators that are likely categorical
        {"pattern": re.compile(r'^TP_'), "reason": "Type indicator - categorical variable"},
        {"pattern": re.compile(r'^IN_'), "reason": "Boolean indicator - categorical variable"}
    ]
    
    return exclude_patterns

def early_feature_reduction(df, sample_size=10000, missing_threshold=0.3, cardinality_threshold=0.95):
    """
    Perform early cuts to reduce the number of columns before multicollinearity analysis
    
    Parameters:
    ----------
    df : pandas DataFrame
        The original dataset
    sample_size : int, default=10000
        Number of rows to sample for faster processing
    missing_threshold : float, default=0.3
        Columns with more than this proportion of missing values will be dropped
    cardinality_threshold : float, default=0.95
        Columns with cardinality ratio (unique values / total values) higher than this will be dropped
        
    Returns:
    -------
    reduced_df : pandas DataFrame
        DataFrame with reduced number of columns
    dropped_columns : dict
        Dictionary with columns dropped and reasons
    """
    print(f"Original dataframe: {df.shape[0]} rows, {df.shape[1]} columns")
    
    # Sample the dataframe for faster processing
    if df.shape[0] > sample_size:
        df_sample = df.sample(sample_size, random_state=42)
        print(f"Working with a sample of {sample_size} rows for initial analysis")
    else:
        df_sample = df
        print(f"Using all {df.shape[0]} rows for initial analysis")
    
    dropped_columns = {
        'dictionary_based': [],
        'constant': [],
        'missing_values': [],
        'high_cardinality': [],
        'non_numeric': []
    }
    
    # 0. Drop columns based on data dictionary patterns
    exclude_patterns = identify_dictionary_based_exclusions()
    
    for col in df_sample.columns:
        for pattern_dict in exclude_patterns:
            if pattern_dict["pattern"].match(col):
                dropped_columns['dictionary_based'].append((col, pattern_dict["reason"]))
                break
    
    # Get column names only from the tuples
    dict_based_cols = [col for col, _ in dropped_columns['dictionary_based']]
    df_sample = df_sample.drop(columns=dict_based_cols)
    print(f"Dropped {len(dict_based_cols)} columns based on data dictionary patterns")
    
    # 1. Drop columns with constant values
    constant_cols = [col for col in df_sample.columns if df_sample[col].nunique() <= 1]
    dropped_columns['constant'] = constant_cols
    df_sample = df_sample.drop(columns=constant_cols)
    print(f"Dropped {len(constant_cols)} constant columns")
    
    # 2. Drop columns with too many missing values
    missing_cols = [col for col in df_sample.columns 
                   if df_sample[col].isna().sum() / df_sample.shape[0] > missing_threshold]
    dropped_columns['missing_values'] = missing_cols
    df_sample = df_sample.drop(columns=missing_cols)
    print(f"Dropped {len(missing_cols)} columns with >{missing_threshold*100}% missing values")
    
    # 3. Remove high cardinality columns (likely IDs or unique identifiers with no analytical value)
    high_cardinality_cols = []
    for col in df_sample.columns:
        if df_sample[col].dtype != 'object':  # Skip non-numeric columns
            n_unique = df_sample[col].nunique()
            if n_unique / df_sample.shape[0] > cardinality_threshold:
                high_cardinality_cols.append(col)
    
    dropped_columns['high_cardinality'] = high_cardinality_cols
    df_sample = df_sample.drop(columns=high_cardinality_cols)
    print(f"Dropped {len(high_cardinality_cols)} high cardinality columns (likely IDs or unique values)")
    
    # 4. Keep only numeric columns for correlation analysis
    numeric_cols = df_sample.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_cols = [col for col in df_sample.columns if col not in numeric_cols]
    dropped_columns['non_numeric'] = non_numeric_cols
    
    df_reduced = df_sample[numeric_cols]
    print(f"Set aside {len(non_numeric_cols)} non-numeric columns")
    print(f"Retained {len(numeric_cols)} numeric columns for correlation analysis")
    
    return df_reduced, dropped_columns

def analyze_correlation_batches(df, batch_size=50, correlation_threshold=0.7):
    """
    Analyze correlation in batches to reduce memory consumption
    
    Parameters:
    ----------
    df : pandas DataFrame
        DataFrame with numeric columns only
    batch_size : int, default=50
        Size of column batches to process at once
    correlation_threshold : float, default=0.7
        Threshold for considering correlations as high
    
    Returns:
    -------
    high_corr_pairs : list
        List of tuples (var1, var2, corr) for highly correlated pairs
    """
    columns = df.columns.tolist()
    n_cols = len(columns)
    
    high_corr_pairs = []
    
    print(f"Analyzing correlations for {n_cols} columns in batches of {batch_size}...")
    
    # Process correlation in batches
    for i in range(0, n_cols, batch_size):
        batch_end = min(i + batch_size, n_cols)
        batch_cols = columns[i:batch_end]
        
        print(f"Processing batch {i//batch_size + 1}: columns {i+1} to {batch_end}")
        
        # Calculate correlation for this batch of columns
        corr_batch = df[batch_cols].corr()
        
        # Find highly correlated pairs within this batch
        for j in range(len(batch_cols)):
            for k in range(j+1, len(batch_cols)):
                if abs(corr_batch.iloc[j, k]) > correlation_threshold:
                    high_corr_pairs.append((
                        batch_cols[j], 
                        batch_cols[k], 
                        corr_batch.iloc[j, k]
                    ))
        
        # If not the last batch, calculate correlation between this batch and remaining columns
        if batch_end < n_cols:
            for next_batch_start in range(batch_end, n_cols, batch_size):
                next_batch_end = min(next_batch_start + batch_size, n_cols)
                next_batch_cols = columns[next_batch_start:next_batch_end]
                
                print(f"  Cross-correlating with columns {next_batch_start+1} to {next_batch_end}")
                
                # Calculate correlation between each column in current batch and next batch
                for col1 in batch_cols:
                    for col2 in next_batch_cols:
                        if col1 != col2:  # Avoid self-correlations
                            corr_value = df[col1].corr(df[col2])
                            if abs(corr_value) > correlation_threshold:
                                high_corr_pairs.append((col1, col2, corr_value))
    
    # Remove potential duplicates
    high_corr_pairs = list(set(high_corr_pairs))
    
    return high_corr_pairs

def calculate_vif_in_batches(df, max_columns_for_vif=30):
    """
    Calculate VIF for a subset of columns to avoid computational issues
    
    Parameters:
    ----------
    df : pandas DataFrame
        DataFrame with numeric columns
    max_columns_for_vif : int, default=30
        Maximum number of columns to use for VIF calculation
    
    Returns:
    -------
    vif_data : pandas DataFrame
        DataFrame with VIF values
    """
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        
        # If we have too many columns, select a subset based on correlation with other columns
        if df.shape[1] > max_columns_for_vif:
            print(f"Too many columns ({df.shape[1]}) for practical VIF calculation.")
            print(f"Selecting the {max_columns_for_vif} most representative columns based on correlation...")
            
            # Calculate mean absolute correlation for each column
            corr_matrix = df.corr().abs()
            mean_corr = corr_matrix.mean(axis=1)
            
            # Select columns with highest mean correlation with other columns
            selected_cols = mean_corr.nlargest(max_columns_for_vif).index.tolist()
            print(f"Selected {len(selected_cols)} columns for VIF calculation")
            
            # Use only selected columns
            X = df[selected_cols].dropna()
        else:
            X = df.dropna()
            selected_cols = df.columns.tolist()
        
        # Check if we have enough data after dropping NAs
        if X.shape[0] < 10:
            print("Not enough data points for VIF calculation after removing NAs.")
            return None
        
        # Calculate VIF for each feature
        vif_data = pd.DataFrame()
        vif_data["Variable"] = X.columns
        
        print("Calculating VIF values...")
        vif_values = []
        for i in range(X.shape[1]):
            vif = variance_inflation_factor(X.values, i)
            vif_values.append(vif)
        
        vif_data["VIF"] = vif_values
        
        # Sort by VIF value
        vif_data = vif_data.sort_values("VIF", ascending=False)
        
        return vif_data
    
    except Exception as e:
        print(f"Error calculating VIF: {e}")
        return None

def recommend_columns_to_drop(high_corr_pairs, correlation_threshold=0.7):
    """
    Recommend columns to drop based on correlation analysis
    
    Parameters:
    ----------
    high_corr_pairs : list
        List of tuples (var1, var2, corr) for highly correlated pairs
    correlation_threshold : float, default=0.7
        Threshold for considering correlations as high
    
    Returns:
    -------
    drop_candidates : list
        List of columns recommended to drop
    """
    if not high_corr_pairs:
        print("No highly correlated pairs found. No recommendations for dropping columns.")
        return []
    
    # Count how many times each variable appears in high correlation pairs
    var_counts = {}
    for var1, var2, _ in high_corr_pairs:
        var_counts[var1] = var_counts.get(var1, 0) + 1
        var_counts[var2] = var_counts.get(var2, 0) + 1
    
    # Group variables by correlation networks
    corr_networks = []
    for var1, var2, _ in high_corr_pairs:
        found = False
        for network in corr_networks:
            if var1 in network or var2 in network:
                network.update([var1, var2])
                found = True
                break
        if not found:
            corr_networks.append(set([var1, var2]))
    
    # Merge overlapping networks
    i = 0
    while i < len(corr_networks):
        j = i + 1
        merged = False
        while j < len(corr_networks):
            if not corr_networks[i].isdisjoint(corr_networks[j]):
                corr_networks[i] = corr_networks[i].union(corr_networks[j])
                corr_networks.pop(j)
                merged = True
            else:
                j += 1
        if not merged:
            i += 1
    
    # For each network, select variables to drop based on their frequency in high correlation pairs
    drop_candidates = []
    for network in corr_networks:
        if len(network) > 1:
            # Sort network variables by frequency in high correlation pairs (descending)
            network_vars = sorted([(var, var_counts[var]) for var in network], 
                                 key=lambda x: x[1], reverse=True)
            
            # Keep the first variable (least correlated with others) and suggest dropping the rest
            vars_to_drop = [var for var, _ in network_vars[1:]]
            drop_candidates.extend(vars_to_drop)
    
    return drop_candidates

def process_large_census_data(file_path, encoding='latin1', separator=';'):
    """
    Main function to process large census data and identify columns to drop
    
    Parameters:
    ----------
    file_path : str
        Path to the data file
    encoding : str, default='latin1'
        File encoding
    separator : str, default=';'
        File delimiter
    """
    start_time = time.time()
    
    # Step 1: Load the data
    print(f"Reading file from {file_path}...")
    df = pd.read_parquet(file_path)

    
    print(f"Data loaded successfully with {df.shape[0]} rows and {df.shape[1]} columns.")
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    
    # Step 2: Early feature reduction
    print("\n--- Performing Early Feature Reduction (Using Dictionary Knowledge) ---")
    reduced_df, dropped_columns = early_feature_reduction(df)
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    
    # Step 3: Analyze correlations in batches
    print("\n--- Analyzing Correlations in Batches ---")
    high_corr_pairs = analyze_correlation_batches(reduced_df)
    
    # Print highly correlated pairs
    print(f"\nFound {len(high_corr_pairs)} highly correlated variable pairs (|r| > 0.7):")
    for var1, var2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:20]:
        print(f"{var1} and {var2}: {corr:.4f}")
    
    if len(high_corr_pairs) > 20:
        print(f"... and {len(high_corr_pairs) - 20} more pairs")
    
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    
    # Step 4: Calculate VIF for a subset of columns
    print("\n--- Calculating VIF for Selected Columns ---")
    vif_data = calculate_vif_in_batches(reduced_df)
    
    if vif_data is not None:
        print("\nVariance Inflation Factor (VIF) Results (top 20):")
        print("VIF > 10 suggests high multicollinearity")
        print("VIF > 5 suggests moderate multicollinearity")
        print(vif_data.head(20))
        
        # Identify variables with high VIF
        high_vif_vars = vif_data[vif_data["VIF"] > 5]["Variable"].tolist()
        
        if high_vif_vars:
            print(f"\nVariables with high multicollinearity (VIF > 5): {len(high_vif_vars)}")
            print(high_vif_vars[:10], "..." if len(high_vif_vars) > 10 else "")
    
    print(f"Time elapsed: {time.time() - start_time:.2f} seconds")
    
    # Step 5: Recommend columns to drop
    print("\n--- Recommending Columns to Drop ---")
    drop_candidates = recommend_columns_to_drop(high_corr_pairs)
    
    # Summarize all recommended columns to drop
    all_drops = {}
    all_drops["dictionary_based"] = [col for col, _ in dropped_columns["dictionary_based"]]
    all_drops["constant_values"] = dropped_columns["constant"]
    all_drops["high_missing_values"] = dropped_columns["missing_values"]
    all_drops["high_cardinality"] = dropped_columns["high_cardinality"]
    all_drops["high_correlation"] = drop_candidates
    all_drops["non_numeric"] = dropped_columns["non_numeric"]
    
    print("\nSummary of all recommended columns to drop:")
    total_drops = 0
    for reason, cols in all_drops.items():
        if cols:
            print(f"- {reason}: {len(cols)} columns")
            print(f"  Examples: {', '.join(cols[:5])}" + ("..." if len(cols) > 5 else ""))
            total_drops += len(cols)
    
    print(f"\nTotal columns recommended to drop: {total_drops}")
    print(f"Original column count: {df.shape[1]}")
    print(f"Potential reduced column count: {df.shape[1] - total_drops}")
    
    # Save recommendations to file
    print("\nSaving drop recommendations to 'column_drop_recommendations.csv'")
    with open('column_drop_recommendations.csv', 'w') as f:
        f.write("reason,column_name\n")
        for reason, cols in all_drops.items():
            for col in cols:
                f.write(f"{reason},{col}\n")
    
    # Create a correlation heatmap for the top correlated variables
    if high_corr_pairs:
        print("\nCreating correlation heatmap for top correlated variables...")
        # Get unique variables from top 30 correlation pairs
        top_pairs = sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True)[:30]
        top_vars = set()
        for var1, var2, _ in top_pairs:
            top_vars.add(var1)
            top_vars.add(var2)
        
        # Limit to at most 30 variables for readability
        top_vars = list(top_vars)[:30]
        
        # Create heatmap
        plt.figure(figsize=(12, 10))
        sns.heatmap(reduced_df[top_vars].corr(), annot=False, cmap='coolwarm', center=0)
        plt.title('Correlation Heatmap for Top Correlated Variables')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('top_correlation_heatmap.png')
        plt.close()
        print("Heatmap saved as 'top_correlation_heatmap.png'")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")
    
    print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    # Prompt user for file path
    file_path = "data/dados_censo/MICRODADOS_CADASTRO_CURSOS_2012_2023.parquet"

    # Process the data
    process_large_census_data(file_path)
    
    print("\nAnalysis complete!")