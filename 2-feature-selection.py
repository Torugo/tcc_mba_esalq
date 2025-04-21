import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def create_output_dirs():
    """Create unique output directories with timestamp."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = 'data/outputs'
    run_dir = f'{base_dir}/run_{timestamp}'
    plots_dir = f'{run_dir}/plots'
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    print(f"Created output directory: {run_dir}")
    return run_dir, plots_dir


def reduce_mem_usage(df):
    """Reduce memory usage of a dataframe by downcasting numeric columns."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object and col_type != 'category':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    
    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df


def add_time_variables(df):
    """Add meaningful time-based variables derived from year fields."""
    time_vars = df.copy()
    
    # Only calculate if both columns exist and are numeric
    if ('Ano de Referência' in df.columns and 'Ano de Ingresso' in df.columns and 
        pd.api.types.is_numeric_dtype(df['Ano de Referência']) and 
        pd.api.types.is_numeric_dtype(df['Ano de Ingresso'])):
        
        # Time since enrollment (in years)
        time_vars['Anos_Desde_Ingresso'] = df['Ano de Referência'] - df['Ano de Ingresso']
        
        # Progress ratio (if integration timeframe exists)
        if 'Prazo de Integralização em Anos' in df.columns and pd.api.types.is_numeric_dtype(df['Prazo de Integralização em Anos']):
            # Avoid division by zero
            valid_idx = df['Prazo de Integralização em Anos'] > 0
            time_vars.loc[valid_idx, 'Progresso_Relativo'] = (
                (df.loc[valid_idx, 'Ano de Referência'] - df.loc[valid_idx, 'Ano de Ingresso']) / 
                df.loc[valid_idx, 'Prazo de Integralização em Anos']
            )
        
        # Time until expected completion (if integration year exists)
        if 'Ano de Integralização do Curso' in df.columns and pd.api.types.is_numeric_dtype(df['Ano de Integralização do Curso']):
            time_vars['Anos_Até_Conclusão'] = df['Ano de Integralização do Curso'] - df['Ano de Referência']
    
    return time_vars


def analyze_correlations(df, target, top_n=20, exclude_vars=None):
    """Calculate correlations with target for all numeric columns."""
    if exclude_vars is None:
        exclude_vars = []
    
    numeric_cols = df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 
                                           'float16', 'float32', 'float64']).columns
    
    # Filter out the target variable, other dropout metrics, and leakage variables
    dropout_cols = ['Taxa de Desistência Acumulada - TDA', 'Taxa de Desistência Anual - TADA']
    numeric_cols = [col for col in numeric_cols if col not in dropout_cols and col not in exclude_vars]
    
    correlation = pd.DataFrame()
    for col in numeric_cols:
        if df[col].nunique() > 1:  # Skip constant columns
            # Only calculate correlation if both columns have valid values
            valid_data = df[[col, target]].dropna()
            if len(valid_data) > 0:
                correlation.loc[col, 'Pearson Correlation'] = valid_data[col].corr(valid_data[target], method='pearson')
                correlation.loc[col, 'Spearman Correlation'] = valid_data[col].corr(valid_data[target], method='spearman')
    
    # Sort by absolute correlation
    correlation['Abs Pearson'] = correlation['Pearson Correlation'].abs()
    top_correlated = correlation.sort_values('Abs Pearson', ascending=False).head(top_n)
    
    return top_correlated


def analyze_mutual_info(df, target, top_n=20, exclude_vars=None):
    """Calculate mutual information between features and target."""
    if exclude_vars is None:
        exclude_vars = []
    
    numeric_cols = df.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 
                                           'float16', 'float32', 'float64']).columns
    
    # Filter out the target variable, other dropout metrics, and leakage variables
    dropout_cols = ['Taxa de Desistência Acumulada - TDA', 'Taxa de Desistência Anual - TADA']
    numeric_cols = [col for col in numeric_cols if col not in dropout_cols and col not in exclude_vars]
    
    # Create X and y for mutual information calculation
    X = df[numeric_cols].copy()
    y = df[target].copy()
    
    # Remove rows with missing target values
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Fill missing values (MI can't handle NaNs)
    X = X.fillna(X.median())
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y)
    mi_df = pd.DataFrame({'Feature': numeric_cols, 'MI Score': mi_scores})
    mi_df = mi_df.sort_values('MI Score', ascending=False).head(top_n)
    
    return mi_df


def analyze_categorical_features(df, target, max_unique=30, exclude_vars=None):
    """Analyze categorical features relationship with target using ANOVA."""
    if exclude_vars is None:
        exclude_vars = []
    
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_cols = [col for col in categorical_cols if col not in exclude_vars and df[col].nunique() <= max_unique]
    
    results = []
    for col in categorical_cols:
        groups = []
        labels = []
        
        # Create groups for ANOVA
        for label, group in df.groupby(col):
            if not group[target].isna().all():
                groups.append(group[target].dropna())
                labels.append(label)
        
        # Run ANOVA if we have at least 2 groups
        if len(groups) >= 2:
            try:
                f_val, p_val = stats.f_oneway(*groups)
                results.append({
                    'Feature': col,
                    'F-statistic': f_val,
                    'p-value': p_val,
                    'Unique Values': df[col].nunique()
                })
            except Exception as e:
                print(f"Error analyzing {col}: {e}")
    
    # Create and sort results dataframe
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values('p-value').reset_index(drop=True)
    
    return results_df


def get_top_features(df, top_correlations, top_mi, cat_features, n_features=30):
    """Combine top features from different methods."""
    # Get top correlated features
    corr_features = top_correlations.index.tolist()
    
    # Get top mutual information features
    mi_features = top_mi['Feature'].tolist()
    
    # Get top categorical features
    if not cat_features.empty and 'Feature' in cat_features.columns:
        cat_features = cat_features.head(10)['Feature'].tolist()
    else:
        cat_features = []
    
    # Combine all features
    all_features = list(set(corr_features + mi_features + cat_features))
    
    # Check if features exist in dataframe and limit to top n
    valid_features = [f for f in all_features if f in df.columns]
    return valid_features[:n_features]


def preprocess_data(X_train, X_test, numeric_cols, categorical_cols):
    """Preprocess data with simple imputation and encoding."""
    # Process numeric features
    if len(numeric_cols) > 0:
        num_imputer = SimpleImputer(strategy='median')
        X_train_num = num_imputer.fit_transform(X_train[numeric_cols])
        X_test_num = num_imputer.transform(X_test[numeric_cols])
        
        # Scale numeric features
        scaler = StandardScaler()
        X_train_num = scaler.fit_transform(X_train_num)
        X_test_num = scaler.transform(X_test_num)
    else:
        X_train_num = np.array([]).reshape(X_train.shape[0], 0)
        X_test_num = np.array([]).reshape(X_test.shape[0], 0)
    
    # Process categorical features with one-hot encoding
    if len(categorical_cols) > 0:
        X_train_cat = pd.get_dummies(X_train[categorical_cols], drop_first=True)
        X_test_cat = pd.get_dummies(X_test[categorical_cols], drop_first=True)
        
        # Ensure X_test has same columns as X_train
        for col in X_train_cat.columns:
            if col not in X_test_cat.columns:
                X_test_cat[col] = 0
        X_test_cat = X_test_cat[X_train_cat.columns]
        
        # Convert to numpy
        X_train_cat = X_train_cat.values
        X_test_cat = X_test_cat.values
    else:
        X_train_cat = np.array([]).reshape(X_train.shape[0], 0)
        X_test_cat = np.array([]).reshape(X_test.shape[0], 0)
    
    # Combine numeric and categorical features
    X_train_proc = np.hstack([X_train_num, X_train_cat])
    X_test_proc = np.hstack([X_test_num, X_test_cat])
    
    return X_train_proc, X_test_proc


def train_evaluate_model(X_train_proc, X_test_proc, y_train, y_test):
    """Train and evaluate a Random Forest model."""
    # Train Random Forest model
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf.fit(X_train_proc, y_train)
    
    # Evaluate model
    y_pred = rf.predict(X_test_proc)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return rf, r2, rmse


def get_feature_importances(rf, feature_names):
    """Extract feature importances from trained model."""
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    features_df = pd.DataFrame({
        'Feature': [feature_names[i] for i in indices],
        'Importance': importances[indices]
    })
    
    return features_df


def analyze_feature_groups(features_df, feature_groups):
    """Calculate average importance for each feature group."""
    # Ensure features_df has Feature column in the right format
    if 'Feature' not in features_df.columns:
        raise ValueError("features_df must have a 'Feature' column")
    
    # Get all feature importance values
    all_features = features_df['Feature'].tolist()
    
    # Calculate importance for each group
    group_importance = {}
    for group_name, group_features in feature_groups.items():
        # Find features from this group in the importance dataframe
        matched_features = []
        for feature in all_features:
            if any(group_feature in feature for group_feature in group_features):
                matched_features.append(feature)
        
        if matched_features:
            # Calculate average importance for this group
            group_df = features_df[features_df['Feature'].isin(matched_features)]
            avg_importance = group_df['Importance'].mean()
            top_features = group_df.head(3)['Feature'].tolist()
            
            group_importance[group_name] = {
                'Average Importance': avg_importance,
                'Top Features': top_features,
                'Matched Features Count': len(matched_features)
            }
    
    # Sort by average importance
    sorted_groups = sorted(group_importance.items(), 
                         key=lambda x: x[1]['Average Importance'], 
                         reverse=True)
    
    return sorted_groups


def create_exploratory_plots(df, target, top_features, plots_dir, max_plots=8):
    """Create exploratory plots for top features."""
    # Get top features (limit to maximum number of plots)
    plot_features = top_features['Feature'].head(max_plots).tolist()
    
    # Create plots for each feature
    for i, feature in enumerate(plot_features):
        if feature in df.columns:
            plt.figure(figsize=(10, 6))
            
            # Check feature type and create appropriate plot
            if df[feature].dtype in ['int8', 'int16', 'int32', 'int64', 
                                   'float16', 'float32', 'float64']:
                # For numeric features, create scatter plot
                plt.scatter(df[feature], df[target], alpha=0.5)
                plt.title(f'Relationship between {feature} and {target}')
                plt.xlabel(feature)
                plt.ylabel(target)
                
                # Add trendline
                if df[[feature, target]].dropna().shape[0] > 1:
                    slope, intercept, r_value, p_value, std_err = stats.linregress(
                        df[feature].dropna(), df[target].dropna())
                    x = np.array([df[feature].min(), df[feature].max()])
                    y = intercept + slope * x
                    plt.plot(x, y, 'r')
                    plt.text(0.05, 0.95, f'R² = {r_value**2:.3f}, p = {p_value:.3f}', 
                           transform=plt.gca().transAxes)
            
            elif df[feature].nunique() <= 10:
                # For categorical features with few categories, create boxplot
                sns.boxplot(x=feature, y=target, data=df)
                plt.title(f'Relationship between {feature} and {target}')
                plt.xticks(rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{plots_dir}/{i+1}_{feature}_vs_{target}.png')
            plt.close()
    
    print(f"Created {min(len(plot_features), max_plots)} exploratory plots in '{plots_dir}/' directory")


def save_results(output_dir, plots_dir, top_correlations, top_mi, cat_features_analysis, features_df, 
                group_importance, r2, rmse, key_vars_analysis, df, df_sample, target):
    """Save all analysis results to output directory."""
    # Save correlation results
    top_correlations.to_csv(f'{output_dir}/top_correlations.csv')
    
    # Save mutual information results
    top_mi.to_csv(f'{output_dir}/top_mutual_info.csv')
    
    # Save categorical features results
    if not cat_features_analysis.empty:
        cat_features_analysis.to_csv(f'{output_dir}/categorical_features_anova.csv')
    
    # Save feature importances
    features_df.to_csv(f'{output_dir}/feature_importances.csv')
    
    # Save model results
    with open(f'{output_dir}/model_results.txt', 'w') as f:
        f.write(f"Random Forest Regressor Results\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances for Dropout Rate Prediction')
    plt.bar(range(min(20, len(features_df))), features_df['Importance'].values[:20], align='center')
    plt.xticks(range(min(20, len(features_df))), features_df['Feature'].values[:20], rotation=90)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importances.png')
    
    # Save feature group analysis
    with open(f'{output_dir}/feature_group_analysis.txt', 'w') as f:
        f.write("Feature groups ranked by importance:\n")
        for group_name, data in group_importance:
            f.write(f"\n{group_name}:\n")
            f.write(f"  Average Importance: {data['Average Importance']:.4f}\n")
            f.write(f"  Matched Features: {data['Matched Features Count']}\n")
            f.write(f"  Top Features: {', '.join(data['Top Features'])}\n")
    
    # Save key variables analysis if provided
    if key_vars_analysis is not None:
        pd.DataFrame(key_vars_analysis, columns=['Variable', 'Correlation', 'Selection Method']).to_csv(
            f'{output_dir}/key_variables_analysis.csv', index=False)
    
    # Save run metadata
    with open(f'{output_dir}/run_info.txt', 'w') as f:
        f.write(f"Run timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total rows in dataset: {len(df)}\n")
        f.write(f"Sample size used: {len(df_sample)}\n")
        f.write(f"Number of features: {df.shape[1]}\n")
        f.write(f"Target variable: {target}\n")
    
    print(f"\nAll analysis results saved to {output_dir}/")


if __name__ == "__main__":
    # Create output directories
    output_dir, plots_dir = create_output_dirs()
    
    # Define leakage variables to exclude
    leakage_variables = [
        'Taxa de Permanência - TAP',
        'Taxa de Conclusão Acumulada - TCA', 
        'Taxa de Conclusão Anual - TCAN',
        'Taxa de Desistência Anual - TADA'
    ]
    
    # Load data with optimized types
    df = pd.read_parquet('data/joined_data.parquet')
    df = reduce_mem_usage(df)
    
    # If needed, use sampling for large datasets
    if len(df) > 100000:
        sample_size = 100000
        df_sample = df.sample(sample_size, random_state=42)
        print(f"Using a sample of {sample_size} rows for initial analysis")
    else:
        df_sample = df
    
    # Add derived time variables
    print("Adding derived time variables...")
    df_sample = add_time_variables(df_sample)
    
    # Define the target variable 
    target = 'Taxa de Desistência Acumulada - TDA'
    
    # Run correlation analysis excluding leakage variables
    top_correlations = analyze_correlations(df_sample, target, exclude_vars=leakage_variables)
    print("\nTop correlated features with dropout rate:")
    print(top_correlations[['Pearson Correlation', 'Spearman Correlation']])
    
    # Run mutual information analysis
    top_mi = analyze_mutual_info(df_sample, target, exclude_vars=leakage_variables)
    print("\nTop features by Mutual Information:")
    print(top_mi)
    
    # Run analysis for categorical features
    cat_features_analysis = analyze_categorical_features(df_sample, target)
    print("\nTop categorical features by ANOVA:")
    print(cat_features_analysis.head(20))
    
    # Get combined top features
    top_features = get_top_features(df_sample, top_correlations, top_mi, cat_features_analysis)
    print(f"\nSelected {len(top_features)} top features for modeling")
    
    # Create training data with selected features
    X = df_sample[top_features].copy()
    y = df_sample[target].copy()
    
    # Remove rows with missing target
    valid_indices = ~y.isna()
    X = X[valid_indices]
    y = y[valid_indices]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Get column types for preprocessing
    numeric_cols = X.select_dtypes(include=['int8', 'int16', 'int32', 'int64', 
                                           'float16', 'float32', 'float64']).columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    
    # Preprocess the data
    X_train_proc, X_test_proc = preprocess_data(X_train, X_test, numeric_cols, categorical_cols)
    
    # Train and evaluate model
    rf, r2, rmse = train_evaluate_model(X_train_proc, X_test_proc, y_train, y_test)
    print(f"\nModel Performance - R² Score: {r2:.4f}, RMSE: {rmse:.4f}")
    
    # Get feature names for importance
    if len(numeric_cols) > 0 and len(categorical_cols) > 0:
        cat_features_ohe = pd.get_dummies(X_train[categorical_cols], drop_first=True).columns.tolist()
        feature_names = numeric_cols.tolist() + cat_features_ohe
    elif len(numeric_cols) > 0:
        feature_names = numeric_cols.tolist()
    else:
        feature_names = pd.get_dummies(X_train[categorical_cols], drop_first=True).columns.tolist()
    
    # Get feature importances
    features_df = get_feature_importances(rf, feature_names)
    print("\nTop 20 features by Random Forest importance:")
    print(features_df.head(20))
    
    # Define feature groups based on educational literature
    feature_groups = {
        'Institutional Characteristics': [
            'TP_CATEGORIA_ADMINISTRATIVA', 'TP_ORGANIZACAO_ACADEMICA', 
            'Categoria Administrativa', 'Organização Acadêmica', 'TP_REDE', 'CO_IES'
        ],
        
        'Course Characteristics': [
            'TP_GRAU_ACADEMICO', 'TP_MODALIDADE_ENSINO', 'Grau Acadêmico', 
            'Modalidade de Ensino', 'Prazo de Integralização em Anos',
            'CO_CINE_AREA_GERAL', 'NO_CINE_AREA_GERAL', 'CO_CINE_ROTULO'
        ],
        
        'Student Demographics': [
            'QT_ING_FEM', 'QT_ING_MASC', 'QT_MAT_FEM', 'QT_MAT_MASC',
            'QT_ING_BRANCA', 'QT_ING_PRETA', 'QT_ING_PARDA',
            'QT_ING_AMARELA', 'QT_ING_INDIGENA', 'QT_ING_CORND'
        ],
        
        'Student Age': [
            'QT_ING_0_17', 'QT_ING_18_24', 'QT_ING_25_29',
            'QT_ING_30_34', 'QT_ING_35_39', 'QT_ING_40_49',
            'QT_ING_50_59', 'QT_ING_60_MAIS'
        ],
        
        'Financial Support': [
            'QT_ING_FINANC', 'QT_MAT_FINANC', 'QT_CONC_FINANC',
            'QT_ING_FIES', 'QT_MAT_FIES', 'QT_CONC_FIES',
            'QT_ING_PROUNII', 'QT_MAT_PROUNII', 'QT_CONC_PROUNII',
            'QT_ING_PROUNIP', 'QT_MAT_PROUNIP', 'QT_CONC_PROUNIP'
        ],
        
        'Previous Education': [
            'QT_ING_PROCESCPUBLICA', 'QT_ING_PROCESCPRIVADA',
            'QT_MAT_PROCESCPUBLICA', 'QT_MAT_PROCESCPRIVADA',
            'QT_ING_RESERVA_VAGA', 'QT_MAT_RESERVA_VAGA'
        ],
        
        'Student Support': [
            'QT_APOIO_SOCIAL', 'QT_ING_APOIO_SOCIAL', 'QT_MAT_APOIO_SOCIAL',
            'QT_ATIV_EXTRACURRICULAR', 'QT_ING_ATIV_EXTRACURRICULAR', 'QT_MAT_ATIV_EXTRACURRICULAR'
        ],
        
        'Time Variables': [
            'Anos_Desde_Ingresso', 'Progresso_Relativo', 'Anos_Até_Conclusão'
        ]
    }
    
    # Analyze importance by category
    group_importance = analyze_feature_groups(features_df, feature_groups)
    print("\nFeature groups ranked by importance:")
    for group_name, data in group_importance:
        print(f"\n{group_name}:")
        print(f"  Average Importance: {data['Average Importance']:.4f}")
        print(f"  Matched Features: {data['Matched Features Count']}")
        print(f"  Top Features: {', '.join(data['Top Features'])}")
    
    # Create exploratory plots for top features
    create_exploratory_plots(df_sample, target, features_df, plots_dir)
    
    # Key features identified from correlation output and literature
    key_variables = [
        # Enrollment and demographic factors
        'QT_MAT_DIURNO',              # Daytime enrollments
        'QT_MAT_18_24',               # Students aged 18-24
        'QT_MAT_FEM',                 # Female students
        
        # Financial aspects
        'IN_GRATUITO',                # Free tuition indicator
        'QT_MAT_PROCESCPRIVADA',      # Students from private schools
        
        # Institutional characteristics (research-supported factors)
        'TP_CATEGORIA_ADMINISTRATIVA', # Administrative category
        'TP_MODALIDADE_ENSINO',       # Teaching modality
        'TP_GRAU_ACADEMICO',          # Academic degree
        
        # Course characteristics
        'Prazo de Integralização em Anos', # Course duration
        'CO_CINE_AREA_GERAL',          # Course area
        
        # Time variables
        'Anos_Desde_Ingresso',
        'Progresso_Relativo',
        'Anos_Até_Conclusão'
    ]
    
    # Filter important variables that exist in the dataset
    important_variables = [var for var in key_variables if var in df_sample.columns]
    
    # Analyze key variables
    key_vars_analysis = []
    print("\nKey variables to focus on based on correlations and research:")
    for var in important_variables:
        if var in top_correlations.index:
            print(f"- {var}: Correlation = {top_correlations.loc[var, 'Pearson Correlation']:.4f}")
            key_vars_analysis.append([var, top_correlations.loc[var, 'Pearson Correlation'], 'Correlation analysis'])
        else:
            print(f"- {var}: Important based on educational research")
            key_vars_analysis.append([var, None, 'Educational research'])
    
    # Save all results
    save_results(output_dir, plots_dir, top_correlations, top_mi, cat_features_analysis, 
                features_df, group_importance, r2, rmse, key_vars_analysis, df, df_sample, target)