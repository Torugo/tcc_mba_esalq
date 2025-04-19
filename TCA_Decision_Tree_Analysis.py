import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats
import gc
import os
import time
import datetime
from joblib import dump, load

# Create output folder with timestamp
def create_output_folder(base_folder="tca_analysis_results"):
    """Create a dedicated output folder with timestamp"""
    # Create timestamp string
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"{base_folder}_{timestamp}"
    
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Created output folder: {folder_name}")
    
    return folder_name

# Memory monitoring function
def print_memory_usage(df=None, message="Current memory usage"):
    """Print current memory usage"""
    import psutil
    
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / 1024 / 1024
    
    print(f"{message}: {memory_mb:.2f} MB")
    
    if df is not None:
        df_memory_mb = df.memory_usage(deep=True).sum() / 1024 / 1024
        print(f"DataFrame memory usage: {df_memory_mb:.2f} MB")

def optimize_dtypes(df):
    """Optimize DataFrame dtypes for memory efficiency"""
    start_time = time.time()
    print("Optimizing data types...")
    print_memory_usage(df, "Memory before optimization")
    
    # List of known categorical columns
    categorical_columns = [
        'Categoria Administrativa', 
        'Orgaização Acadêmica',
        'Grau Acadêmico',
        'Modalidade de Ensino',
        'Nome da Grande Área do Curso segundo a classificação CINE BRASIL',
        'SG_UF',
        'NO_REGIAO',
        'NO_CINE_ROTULO',
        'NO_CINE_AREA_GERAL'
    ]
    
    # Convert categorical columns
    for col in categorical_columns:
        if col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype('category')
    
    # Downcast numeric columns
    int_columns = df.select_dtypes(include=['int']).columns
    for col in int_columns:
        if df[col].min() >= 0:  # Unsigned int
            if df[col].max() < 2**8:
                df[col] = df[col].astype(np.uint8)
            elif df[col].max() < 2**16:
                df[col] = df[col].astype(np.uint16)
            elif df[col].max() < 2**32:
                df[col] = df[col].astype(np.uint32)
            else:
                pass  # Keep as is
        else:  # Signed int
            if df[col].min() >= -2**7 and df[col].max() < 2**7:
                df[col] = df[col].astype(np.int8)
            elif df[col].min() >= -2**15 and df[col].max() < 2**15:
                df[col] = df[col].astype(np.int16)
            elif df[col].min() >= -2**31 and df[col].max() < 2**31:
                df[col] = df[col].astype(np.int32)
            else:
                pass  # Keep as is
    
    # Downcast float columns
    float_columns = df.select_dtypes(include=['float']).columns
    for col in float_columns:
        df[col] = df[col].astype(np.float32)
    
    # Look for other object columns with low cardinality
    object_columns = df.select_dtypes(include=['object']).columns
    for col in object_columns:
        if col not in categorical_columns:
            # Only convert if cardinality is low
            n_unique = df[col].nunique()
            if n_unique < len(df) * 0.05:  # If less than 5% unique values
                df[col] = df[col].astype('category')
    
    print_memory_usage(df, "Memory after optimization")
    print(f"Optimization completed in {time.time() - start_time:.2f} seconds")
    return df

def load_data(file_path, sample_size=None, selected_columns=None):
    """Load parquet data with memory optimizations"""
    start_time = time.time()
    print(f"Loading data from {file_path}...")
    
    # Define minimal columns to load if not specifically provided
    if selected_columns is None:
        # Predefine which columns we need to avoid loading all 235
        selected_columns = [
            # Target
            'Taxa de Conclusão Acumulada - TCA',
            
            # Institution characteristics
            'Código da Instituição', 
            'Categoria Administrativa',
            'Organização Acadêmica',
            'TP_CATEGORIA_ADMINISTRATIVA',
            'TP_ORGANIZACAO_ACADEMICA',
            
            # Course characteristics
            'Código do Curso de Graduação',
            'Grau Acadêmico',
            'Modalidade de Ensino',
            'Nome da Grande Área do Curso segundo a classificação CINE BRASIL',
            'TP_GRAU_ACADEMICO',
            'TP_MODALIDADE_ENSINO',
            'Prazo de Integralização em Anos',
            
            # Geographic information
            'Código da Região Geográfica do Curso',
            'Código da Unidade Federativa do Curso',
            'SG_UF',
            'NO_REGIAO',
            
            # Student demographics
            'Quantidade de Ingressantes no Curso',
            'QT_MAT',
            'QT_MAT_FEM',
            'QT_MAT_MASC',
            'QT_MAT_DIURNO',
            'QT_MAT_NOTURNO',
            
            # Age brackets
            'QT_MAT_18_24',
            'QT_MAT_25_29',
            'QT_MAT_30_34',
            'QT_MAT_35_39',
            'QT_MAT_40_49',
            
            # Race/ethnicity
            'QT_MAT_BRANCA',
            'QT_MAT_PRETA',
            'QT_MAT_PARDA',
            
            # Financial aid
            'QT_MAT_FINANC',
            'QT_MAT_FIES',
            'QT_MAT_PROUNII',
            'QT_MAT_PROUNIP',
            
            # School origin
            'QT_MAT_PROCESCPUBLICA',
            'QT_MAT_PROCESCPRIVADA'
        ]
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # If sampling, load only required number of rows
    if sample_size is not None:
        # For parquet, load just the required columns first to check total rows
        print(f"Loading metadata...")
        df_count = pd.read_parquet(file_path, columns=['Taxa de Conclusão Acumulada - TCA'])
        total_rows = len(df_count)
        del df_count
        gc.collect()
        
        if sample_size < total_rows:
            # Use fractional sampling for greater efficiency
            fraction = sample_size / total_rows
            try:
                df = pd.read_parquet(file_path, columns=selected_columns)
                df = df.sample(frac=fraction, random_state=42)
                print(f"Loaded {len(df)} rows as a {fraction:.2%} sample")
            except Exception as e:
                print(f"Error during fractional sampling: {e}")
                # Fallback: read all and then sample
                df = pd.read_parquet(file_path, columns=selected_columns)
                df = df.sample(n=sample_size, random_state=42)
                print(f"Loaded {len(df)} rows using fallback sampling")
        else:
            df = pd.read_parquet(file_path, columns=selected_columns)
            print(f"Loaded all {len(df)} rows (sample size >= total rows)")
    else:
        # Load all data with selected columns
        df = pd.read_parquet(file_path, columns=selected_columns)
        print(f"Loaded all {len(df)} rows with {len(selected_columns)} columns")
    
    # Optimize data types
    df = optimize_dtypes(df)
    
    # Clean up memory
    gc.collect()
    
    print(f"Data loading completed in {time.time() - start_time:.2f} seconds")
    return df

def select_and_prepare_features(df, target_col='Taxa de Conclusão Acumulada - TCA'):
    """Select and prepare features for modeling"""
    start_time = time.time()
    print("\nPreparing features...")
    
    # Define features based on domain knowledge
    categorical_features = [
        'Categoria Administrativa',
        'Organização Acadêmica',
        'Grau Acadêmico',
        'Modalidade de Ensino',
        'Nome da Grande Área do Curso segundo a classificação CINE BRASIL',
        'SG_UF',
        'NO_REGIAO'
    ]
    
    # Filter out non-existent columns
    categorical_features = [col for col in categorical_features if col in df.columns]
    
    numeric_features = [
        'Prazo de Integralização em Anos',
        'Quantidade de Ingressantes no Curso',
        'QT_MAT',
        'QT_MAT_FEM',
        'QT_MAT_MASC',
        'QT_MAT_DIURNO',
        'QT_MAT_NOTURNO',
        'QT_MAT_18_24',
        'QT_MAT_25_29',
        'QT_MAT_30_34',
        'QT_MAT_35_39',
        'QT_MAT_40_49',
        'QT_MAT_BRANCA',
        'QT_MAT_PRETA',
        'QT_MAT_PARDA',
        'QT_MAT_FINANC',
        'QT_MAT_FIES',
        'QT_MAT_PROUNII',
        'QT_MAT_PROUNIP',
        'QT_MAT_PROCESCPUBLICA',
        'QT_MAT_PROCESCPRIVADA'
    ]
    
    # Filter out non-existent columns
    numeric_features = [col for col in numeric_features if col in df.columns]
    
    print(f"Selected {len(categorical_features)} categorical and {len(numeric_features)} numeric features")
    
    # Create feature matrix and target vector
    features = categorical_features + numeric_features
    X = df[features].copy()
    y = df[target_col].copy()
    
    # Handle missing values
    for col in numeric_features:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].median())
    
    for col in categorical_features:
        if X[col].isnull().sum() > 0:
            X[col] = X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else 'Unknown')
    
    # Memory-efficient label encoding for categorical features
    # This avoids the memory explosion from one-hot encoding
    for col in categorical_features:
        # Handle non-category columns
        if not pd.api.types.is_categorical_dtype(X[col]):
            X[col] = X[col].astype('category')
        
        # Extract codes (same as LabelEncoder but more memory efficient)
        X[col] = X[col].cat.codes.astype('int16')
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Feature preparation completed in {time.time() - start_time:.2f} seconds")
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Clean up memory
    gc.collect()
    
    return X_train, X_test, y_train, y_test, features, categorical_features, numeric_features

def train_decision_tree(X_train, y_train, output_folder, max_depth=8, min_samples_leaf=50):
    """Train a memory-efficient decision tree model"""
    start_time = time.time()
    print("\nTraining decision tree model...")
    
    # Use a simple decision tree with controlled complexity
    model = DecisionTreeRegressor(
        max_depth=max_depth,
        min_samples_split=min_samples_leaf*2,  # Conservative split criterion
        min_samples_leaf=min_samples_leaf,     # Avoid overly specific leaves
        random_state=42
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    print(f"Model training completed in {time.time() - start_time:.2f} seconds")
    
    # Save model to disk to conserve memory
    dump(model, os.path.join(output_folder, 'tca_decision_tree_model.joblib'))
    
    return model

def evaluate_model(model, X_test, y_test, output_folder):
    """Evaluate the trained model"""
    start_time = time.time()
    print("\nEvaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mse)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    
    # Create residual plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, y_test - y_pred, alpha=0.5)
    plt.hlines(y=0, xmin=0, xmax=100, colors='red', linestyles='--')
    plt.xlabel('Predicted TCA')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'residual_plot.png'))
    plt.close()
    
    # Create actual vs predicted plot
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([0, 100], [0, 100], 'r--')
    plt.xlabel('Actual TCA')
    plt.ylabel('Predicted TCA')
    plt.title('Actual vs Predicted TCA')
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'actual_vs_predicted.png'))
    plt.close()
    
    print(f"Evaluation completed in {time.time() - start_time:.2f} seconds")
    
    return {
        'mse': mse,
        'r2': r2,
        'rmse': rmse
    }

def analyze_feature_importance(model, features, output_folder):
    """Analyze and visualize feature importance"""
    start_time = time.time()
    print("\nAnalyzing feature importance...")
    
    # Extract feature importance
    importances = model.feature_importances_
    
    # Sort features by importance
    indices = np.argsort(importances)[::-1]
    sorted_features = np.array(features)[indices]
    sorted_importances = importances[indices]
    
    # Print top features
    print("\nTop 10 features by importance:")
    for i in range(min(10, len(features))):
        print(f"{i+1}. {sorted_features[i]}: {sorted_importances[i]:.4f}")
    
    # Plot feature importance (top 20)
    plt.figure(figsize=(12, 10))
    n_features = min(20, len(features))
    plt.barh(range(n_features), sorted_importances[:n_features], align='center')
    plt.yticks(range(n_features), sorted_features[:n_features])
    plt.xlabel('Feature Importance')
    plt.title('Top Features Affecting Student Completion Rates (TCA)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, 'feature_importance.png'))
    plt.close()
    
    print(f"Feature importance analysis completed in {time.time() - start_time:.2f} seconds")
    
    # Create importance mapping
    importance_mapping = dict(zip(sorted_features, sorted_importances))
    
    return sorted_features, sorted_importances, importance_mapping

def statistical_analysis(df, importance_mapping, output_folder, target_col='Taxa de Conclusão Acumulada - TCA', top_n=10):
    """Perform statistical analysis on top features"""
    start_time = time.time()
    print("\nPerforming statistical analysis...")
    
    results = {}
    feature_count = 0
    
    # Get original column names for categorical features
    cat_cols = df.select_dtypes(include=['category']).columns.tolist()
    
    # Loop through top features
    for feature, importance in importance_mapping.items():
        if feature_count >= top_n:
            break
            
        if feature not in df.columns:
            continue
            
        feature_count += 1
        print(f"\nAnalyzing feature: {feature} (Importance: {importance:.4f})")
        
        # Check feature type
        is_categorical = feature in cat_cols or pd.api.types.is_categorical_dtype(df[feature])
        
        # For categorical features
        if is_categorical:
            try:
                # Prepare data for analysis
                feature_data = df[[feature, target_col]].dropna()
                
                # Calculate average TCA by category
                category_means = feature_data.groupby(feature)[target_col].mean()
                
                # Print category means
                print("Category means:")
                for category, mean in category_means.items():
                    print(f"  {category}: {mean:.2f}")
                
                # ANOVA - only if more than one group with data
                groups = []
                for category in feature_data[feature].unique():
                    group = feature_data[feature_data[feature] == category][target_col].values
                    if len(group) > 0:
                        groups.append(group)
                
                p_val = None
                f_stat = None
                if len(groups) > 1:
                    f_stat, p_val = stats.f_oneway(*groups)
                    print(f"ANOVA: F={f_stat:.4f}, p={p_val:.4f}")
                
                results[feature] = {
                    'type': 'categorical',
                    'importance': importance,
                    'category_means': category_means.to_dict(),
                    'f_stat': f_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05 if p_val is not None else None
                }
                
                # Create plot if not too many categories
                if len(category_means) <= 15:
                    plt.figure(figsize=(12, 6))
                    category_means.sort_values().plot(kind='barh')
                    plt.xlabel('Average TCA')
                    plt.title(f'Average TCA by {feature}')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_folder, f'{feature.replace(" ", "_")}_analysis.png'))
                    plt.close()
                
            except Exception as e:
                print(f"Error analyzing categorical feature: {e}")
        
        # For numeric features
        else:
            try:
                # Prepare data for analysis
                feature_data = df[[feature, target_col]].dropna()
                
                # Calculate correlation
                corr = feature_data[feature].corr(feature_data[target_col])
                print(f"Correlation with TCA: {corr:.4f}")
                
                # T-test (high vs low groups)
                median = feature_data[feature].median()
                high_group = feature_data[feature_data[feature] > median][target_col]
                low_group = feature_data[feature_data[feature] <= median][target_col]
                
                t_stat, p_val = stats.ttest_ind(high_group, low_group, equal_var=False)
                print(f"T-test: t={t_stat:.4f}, p={p_val:.4f}")
                
                results[feature] = {
                    'type': 'numeric',
                    'importance': importance,
                    'correlation': corr,
                    't_stat': t_stat,
                    'p_value': p_val,
                    'significant': p_val < 0.05,
                    'mean_high': high_group.mean(),
                    'mean_low': low_group.mean(),
                    'median': median
                }
                
                # Create scatter plot
                plt.figure(figsize=(10, 6))
                plt.scatter(feature_data[feature], feature_data[target_col], alpha=0.3)
                plt.xlabel(feature)
                plt.ylabel('TCA')
                plt.title(f'Relationship between {feature} and TCA')
                plt.tight_layout()
                plt.savefig(os.path.join(output_folder, f'{feature.replace(" ", "_")}_scatter.png'))
                plt.close()
                
            except Exception as e:
                print(f"Error analyzing numeric feature: {e}")
    
    print(f"Statistical analysis completed in {time.time() - start_time:.2f} seconds")
    return results

def generate_report(importance_mapping, stats_results, metrics, categorical_features, 
                   numeric_features, output_folder):
    """Generate a comprehensive report of findings"""
    start_time = time.time()
    print("\nGenerating analysis report...")
    
    output_path = os.path.join(output_folder, 'tca_analysis_report.md')
    
    lines = []
    lines.append("# TCA (Taxa de Conclusão Acumulada) Analysis Report")
    lines.append("\n## Model Performance")
    lines.append(f"- R² Score: {metrics['r2']:.4f}")
    lines.append(f"- Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    
    # Top features section
    lines.append("\n## Key Factors Influencing Completion Rates")
    lines.append("\nThe decision tree analysis identified the following factors as most influential for predicting completion rates:")
    
    lines.append("\n| Rank | Feature | Importance | Statistical Significance |")
    lines.append("| ---- | ------- | ---------- | ------------------------- |")
    
    rank = 1
    for feature, importance in importance_mapping.items():
        if rank > 10:
            break
            
        stat_info = "Not analyzed"
        if feature in stats_results:
            result = stats_results[feature]
            if result['type'] == 'numeric':
                sig_mark = "✓" if result.get('significant', False) else "✗"
                stat_info = f"r={result['correlation']:.3f}, p={result['p_value']:.4f} {sig_mark}"
            else:
                if result.get('p_value') is not None:
                    sig_mark = "✓" if result.get('significant', False) else "✗"
                    stat_info = f"F={result['f_stat']:.2f}, p={result['p_value']:.4f} {sig_mark}"
                else:
                    stat_info = "Categories differ in means"
        
        lines.append(f"| {rank} | {feature} | {importance:.4f} | {stat_info} |")
        rank += 1
    
    # More detailed analysis of key factors
    lines.append("\n## Detailed Analysis of Key Factors")
    
    # Group factors by type
    institutional_factors = []
    course_factors = []
    student_factors = []
    
    # Categorize factors
    for feature in list(importance_mapping.keys())[:10]:
        if feature in ['Categoria Administrativa', 'Organização Acadêmica']:
            institutional_factors.append(feature)
        elif feature in ['Grau Acadêmico', 'Modalidade de Ensino', 'Nome da Grande Área do Curso segundo a classificação CINE BRASIL', 'Prazo de Integralização em Anos']:
            course_factors.append(feature)
        else:
            student_factors.append(feature)
    
    # Report on institutional factors
    if institutional_factors:
        lines.append("\n### Institutional Factors")
        for factor in institutional_factors:
            if factor in stats_results:
                result = stats_results[factor]
                lines.append(f"\n#### {factor}")
                
                if result['type'] == 'categorical':
                    lines.append("\nAverage completion rates by category:")
                    for category, mean in result['category_means'].items():
                        lines.append(f"- {category}: {mean:.2f}%")
                    
                    if result.get('p_value') is not None:
                        sig_text = "statistically significant" if result.get('significant', False) else "not statistically significant"
                        lines.append(f"\nThe differences between categories are {sig_text} (p={result['p_value']:.4f}).")
    
    # Report on course factors
    if course_factors:
        lines.append("\n### Course Factors")
        for factor in course_factors:
            if factor in stats_results:
                result = stats_results[factor]
                lines.append(f"\n#### {factor}")
                
                if result['type'] == 'categorical':
                    lines.append("\nAverage completion rates by category:")
                    for category, mean in result['category_means'].items():
                        lines.append(f"- {category}: {mean:.2f}%")
                    
                    if result.get('p_value') is not None:
                        sig_text = "statistically significant" if result.get('significant', False) else "not statistically significant"
                        lines.append(f"\nThe differences between categories are {sig_text} (p={result['p_value']:.4f}).")
                else:
                    lines.append(f"\nCorrelation with completion rate: {result['correlation']:.3f}")
                    lines.append(f"Programs with {factor} above the median value ({result['median']:.2f}) have an average completion rate of {result['mean_high']:.2f}%, compared to {result['mean_low']:.2f}% for those below the median.")
    
    # Report on student factors
    if student_factors:
        lines.append("\n### Student Demographic Factors")
        for factor in student_factors:
            if factor in stats_results:
                result = stats_results[factor]
                lines.append(f"\n#### {factor}")
                
                if result['type'] == 'numeric':
                    lines.append(f"\nCorrelation with completion rate: {result['correlation']:.3f}")
                    lines.append(f"Programs with {factor} above the median value ({result['median']:.2f}) have an average completion rate of {result['mean_high']:.2f}%, compared to {result['mean_low']:.2f}% for those below the median.")
    
    # Recommendations
    lines.append("\n## Recommendations")
    lines.append("\nBased on the analysis, the following strategies may help improve student completion rates:")
    
    # Will be filled based on actual results
    lines.append("\n1. [Will be filled based on actual analysis results]")
    lines.append("2. [Will be filled based on actual analysis results]")
    lines.append("3. [Will be filled based on actual analysis results]")
    
    # Methodology
    lines.append("\n## Methodology")
    lines.append("\nThis analysis used a decision tree regression model to identify factors influencing student completion rates (TCA) in higher education.")
    
    # Write report to file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Report generated and saved to {output_path}")
    print(f"Report generation completed in {time.time() - start_time:.2f} seconds")

def main(file_path, sample_size=None, max_depth=8):
    """Main analysis workflow with memory optimization"""
    overall_start_time = time.time()
    print("=== TCA Analysis: Identifying Factors Affecting Completion Rates ===\n")
    
    try:
        # Create output folder with timestamp
        output_folder = create_output_folder()
        
        # 1. Load and prepare data
        df = load_data(file_path, sample_size=sample_size)
        print_memory_usage(message="Memory after data loading")
        
        # 2. Feature preparation
        X_train, X_test, y_train, y_test, features, categorical_features, numeric_features = select_and_prepare_features(df)
        print_memory_usage(message="Memory after feature preparation")
        
        # 3. Train model
        model = train_decision_tree(X_train, y_train, output_folder, max_depth=max_depth)
        print_memory_usage(message="Memory after model training")
        
        # 4. Evaluate model
        metrics = evaluate_model(model, X_test, y_test, output_folder)
        
        # 5. Analyze feature importance
        sorted_features, sorted_importances, importance_mapping = analyze_feature_importance(model, features, output_folder)
        
        # 6. Statistical analysis - use a sample of data if full dataset is too large
        if len(df) > 50000:
            analysis_sample = df.sample(50000, random_state=42)
            print(f"Using 50,000 row sample for statistical analysis")
        else:
            analysis_sample = df
            
        stats_results = statistical_analysis(analysis_sample, importance_mapping, output_folder)
        
        # 7. Generate report
        generate_report(importance_mapping, stats_results, metrics, categorical_features, numeric_features, output_folder)
        
        # 8. Final summary
        print("\n=== Analysis Summary ===")
        print(f"Model performance (R²): {metrics['r2']:.4f}")
        print("\nTop 5 factors affecting completion rates:")
        for i, (feature, importance) in enumerate(list(importance_mapping.items())[:5]):
            print(f"{i+1}. {feature}: {importance:.4f}")
        
        print(f"\nTotal analysis time: {time.time() - overall_start_time:.2f} seconds")
        print(f"\nAll results have been saved to folder: {output_folder}")
        print("Results include:")
        print("- feature_importance.png (Feature importance chart)")
        print("- tca_analysis_report.md (Comprehensive report)")
        print("- residual_plot.png and actual_vs_predicted.png (Model diagnostics)")
        print("- Individual feature analysis plots")
        print("- Trained model saved as tca_decision_tree_model.joblib")
        
    except Exception as e:
        print(f"\nError in analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # File path
    file_path = "joined_data.parquet"
    
    # Adjust these parameters based on your system
    sample_size = 100000  # Set to None to use entire dataset, or lower for memory constraints
    max_depth = 8         # Reducing this value will use less memory
    
    main(file_path, sample_size, max_depth)