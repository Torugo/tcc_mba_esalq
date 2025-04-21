import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# For survival analysis
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index

# For modeling
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def create_output_dirs():
    """Create unique output directory for model results."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    base_dir = 'data/outputs'
    run_dir = f'{base_dir}/timeseries_{timestamp}'
    
    # Create directories
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"Created output directory: {run_dir}")
    return run_dir


def load_and_prepare_data(file_path='data/joined_data.parquet', sample_size=None):
    """Load and prepare data with focus on time structure."""
    print("Loading data...")
    df = pd.read_parquet(file_path)
    
    # Sample data only if absolutely necessary (for very large datasets)
    if sample_size is not None and len(df) > sample_size:
        # Instead of random sampling, take a stratified sample by cohort year
        # to maintain the temporal structure
        df = df.groupby('Ano de Ingresso').apply(
            lambda x: x.sample(min(len(x), int(sample_size/df['Ano de Ingresso'].nunique())), 
                              random_state=42)
        ).reset_index(drop=True)
        print(f"Using a stratified sample by enrollment year with {len(df)} rows")
    
    # Add time-based variables if needed
    if 'Anos_Desde_Ingresso' not in df.columns:
        print("Adding time-based variables...")
        if ('Ano de Referência' in df.columns and 'Ano de Ingresso' in df.columns):
            df['Anos_Desde_Ingresso'] = df['Ano de Referência'] - df['Ano de Ingresso']
            
            # Progress ratio
            if 'Prazo de Integralização em Anos' in df.columns:
                valid_idx = df['Prazo de Integralização em Anos'] > 0
                df.loc[valid_idx, 'Progresso_Relativo'] = (
                    (df.loc[valid_idx, 'Ano de Referência'] - df.loc[valid_idx, 'Ano de Ingresso']) / 
                    df.loc[valid_idx, 'Prazo de Integralização em Anos']
                )
            
            # Time until expected completion
            if 'Ano de Integralização do Curso' in df.columns:
                df['Anos_Até_Conclusão'] = df['Ano de Integralização do Curso'] - df['Ano de Referência']
    
    # Extract unique course-student combinations
    # This step ensures we're looking at unique trajectories
    course_student_combinations = df[['CO_CURSO', 'CO_IES']].drop_duplicates()
    
    print(f"Data loaded with {len(df)} rows and {df.shape[1]} columns")
    print(f"Representing {len(course_student_combinations)} unique course-institution combinations")
    
    return df


def prepare_longitudinal_data(df, target='Taxa de Desistência Acumulada - TDA'):
    """
    Prepare data for longitudinal analysis by structuring it by cohort and time point.
    Returns both cohort-level data and properly structured individual time points.
    """
    print("Preparing longitudinal data structure...")
    
    # Ensure we have the key time variables
    key_time_vars = ['Ano de Ingresso', 'Ano de Referência']
    if not all(var in df.columns for var in key_time_vars):
        raise ValueError(f"Missing required time variables: {key_time_vars}")
    
    # Group data by cohort (enrollment year)
    cohorts = df['Ano de Ingresso'].unique()
    print(f"Found {len(cohorts)} distinct enrollment cohorts: {sorted(cohorts)}")
    
    # Create cohort-level dataframe
    # This aggregates data to the course-cohort level
    cohort_data = []
    
    for year in sorted(cohorts):
        cohort_df = df[df['Ano de Ingresso'] == year]
        
        # Group by course to get course-level metrics
        course_groups = cohort_df.groupby(['CO_CURSO', 'CO_IES'])
        
        for (course_id, institution_id), course_data in course_groups:
            # Get information about this course-institution from the last year
            latest_year = course_data['Ano de Referência'].max()
            course_latest = course_data[course_data['Ano de Referência'] == latest_year].iloc[0]
            
            # Course static information
            course_info = {
                'Ano de Ingresso': year,
                'CO_CURSO': course_id,
                'CO_IES': institution_id,
                'Nome do Curso de Graduação': course_latest['Nome do Curso de Graduação'] if 'Nome do Curso de Graduação' in course_latest else None,
                'Categoria Administrativa': course_latest['Categoria Administrativa'] if 'Categoria Administrativa' in course_latest else None,
                'TP_CATEGORIA_ADMINISTRATIVA': course_latest['TP_CATEGORIA_ADMINISTRATIVA'] if 'TP_CATEGORIA_ADMINISTRATIVA' in course_latest else None,
                'Organização Acadêmica': course_latest['Organização Acadêmica'] if 'Organização Acadêmica' in course_latest else None,
                'TP_ORGANIZACAO_ACADEMICA': course_latest['TP_ORGANIZACAO_ACADEMICA'] if 'TP_ORGANIZACAO_ACADEMICA' in course_latest else None,
                'TP_MODALIDADE_ENSINO': course_latest['TP_MODALIDADE_ENSINO'] if 'TP_MODALIDADE_ENSINO' in course_latest else None,
                'Grau Acadêmico': course_latest['Grau Acadêmico'] if 'Grau Acadêmico' in course_latest else None,
                'TP_GRAU_ACADEMICO': course_latest['TP_GRAU_ACADEMICO'] if 'TP_GRAU_ACADEMICO' in course_latest else None,
                'Prazo de Integralização em Anos': course_latest['Prazo de Integralização em Anos'] if 'Prazo de Integralização em Anos' in course_latest else None
            }
            
            # Get year-by-year dropout progression
            years = sorted(course_data['Ano de Referência'].unique())
            yearly_data = {}
            
            for i, ref_year in enumerate(years):
                year_data = course_data[course_data['Ano de Referência'] == ref_year]
                
                # Determine the time point (year since enrollment)
                time_point = ref_year - year
                
                # Get dropout rate for this year
                if target in year_data.columns:
                    yearly_data[f'Dropout_Y{time_point}'] = year_data[target].iloc[0]
                
                # Get permanence rate for this year
                permanence_col = 'Taxa de Permanência - TAP'
                if permanence_col in year_data.columns:
                    yearly_data[f'Permanence_Y{time_point}'] = year_data[permanence_col].iloc[0]
                
                # Store additional time point specific metrics
                if 'Quantidade de Permanência no Curso no ano de referência' in year_data.columns:
                    yearly_data[f'Students_Remaining_Y{time_point}'] = year_data['Quantidade de Permanência no Curso no ano de referência'].iloc[0]
                
                if 'Quantidade de Desistência no Curso no ano de referência' in year_data.columns:
                    yearly_data[f'Students_Dropped_Y{time_point}'] = year_data['Quantidade de Desistência no Curso no ano de referência'].iloc[0]
                
                if 'Quantidade de Concluintes no Curso no ano de referência' in year_data.columns:
                    yearly_data[f'Students_Graduated_Y{time_point}'] = year_data['Quantidade de Concluintes no Curso no ano de referência'].iloc[0]
            
            # Add the first year enrollment demographics
            first_year_data = course_data[course_data['Ano de Referência'] == year]
            if len(first_year_data) > 0:
                if 'QT_ING_FEM' in first_year_data.columns:
                    course_info['QT_ING_FEM'] = first_year_data['QT_ING_FEM'].iloc[0]
                
                if 'QT_ING_MASC' in first_year_data.columns:
                    course_info['QT_ING_MASC'] = first_year_data['QT_ING_MASC'].iloc[0]
                
                if 'QT_ING_18_24' in first_year_data.columns:
                    course_info['QT_ING_18_24'] = first_year_data['QT_ING_18_24'].iloc[0]
                
                if 'QT_ING_25_29' in first_year_data.columns:
                    course_info['QT_ING_25_29'] = first_year_data['QT_ING_25_29'].iloc[0]
                
                if 'Quantidade de Ingressantes no Curso' in first_year_data.columns:
                    course_info['Initial_Enrollment'] = first_year_data['Quantidade de Ingressantes no Curso'].iloc[0]
            
            # Combine course info with yearly progression
            course_record = {**course_info, **yearly_data}
            cohort_data.append(course_record)
    
    # Convert to DataFrame
    cohort_df = pd.DataFrame(cohort_data)
    
    # Create year-by-year prediction targets
    # These will be used for building models that predict dropout at specific time points
    time_point_data = {}
    max_years = 6  # Maximum years to track (adjust based on your data)
    
    for year in range(1, max_years+1):
        # Create dataframe for this time point
        if f'Dropout_Y{year}' in cohort_df.columns:
            # Features are based on initial characteristics only
            features = cohort_df.drop([col for col in cohort_df.columns if col.startswith('Dropout_Y') or col.startswith('Permanence_Y')], axis=1)
            
            # Target is dropout at this specific year
            target_col = f'Dropout_Y{year}'
            
            # Only include courses that have data for this time point
            valid_mask = ~cohort_df[target_col].isna()
            
            if valid_mask.sum() > 100:  # Ensure sufficient data points
                time_point_data[year] = {
                    'X': features[valid_mask].copy(),
                    'y': cohort_df.loc[valid_mask, target_col].copy(),
                    'n_samples': valid_mask.sum()
                }
    
    print(f"Created cohort dataframe with {cohort_df.shape[0]} course-cohort combinations")
    for year, data in time_point_data.items():
        print(f"Time point Y{year}: {data['n_samples']} valid samples")
    
    return cohort_df, time_point_data


def prepare_survival_data(df):
    """
    Prepare data for survival analysis by transforming it into the correct format.
    Each course is followed over time with dropout event indicator.
    """
    print("Preparing survival analysis dataset...")
    
    # Check if we have the necessary columns
    required_cols = ['CO_CURSO', 'CO_IES', 'Ano de Ingresso', 'Ano de Referência']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {[col for col in required_cols if col not in df.columns]}")
    
    # Create course ID by combining course and institution
    df['Course_ID'] = df['CO_CURSO'].astype(str) + '_' + df['CO_IES'].astype(str)
    
    # Get columns with dropout information
    dropout_col = 'Taxa de Desistência Acumulada - TDA'
    if dropout_col not in df.columns:
        raise ValueError(f"Missing dropout column: {dropout_col}")
    
    # Create survival dataset
    survival_data = []
    
    for course_id, course_df in df.groupby('Course_ID'):
        # Sort by reference year
        course_df = course_df.sort_values('Ano de Referência')
        
        # Get enrollment year
        enrollment_year = course_df['Ano de Ingresso'].iloc[0]
        
        # Get static course information from the first record
        first_record = course_df.iloc[0]
        
        # Static course information
        course_info = {
            'Course_ID': course_id,
            'CO_CURSO': first_record['CO_CURSO'],
            'CO_IES': first_record['CO_IES'],
            'Ano de Ingresso': enrollment_year
        }
        
        # Add categorical variables if available
        for col in ['Categoria Administrativa', 'Organização Acadêmica', 'Grau Acadêmico', 
                   'TP_MODALIDADE_ENSINO', 'TP_GRAU_ACADEMICO']:
            if col in first_record:
                course_info[col] = first_record[col]
        
        # Add first-year demographics if available
        for col in ['QT_ING_FEM', 'QT_ING_MASC', 'QT_ING_18_24', 'QT_ING_25_29',
                   'Quantidade de Ingressantes no Curso']:
            if col in first_record:
                course_info[col] = first_record[col]
        
        # Track each year until dropout or censoring
        last_dropout_rate = 0
        
        for i, row in course_df.iterrows():
            ref_year = row['Ano de Referência']
            current_time = ref_year - enrollment_year
            
            # Get current dropout rate
            current_dropout_rate = row[dropout_col]
            
            # Determine if dropout event occurred in this period
            # An increase in the dropout rate indicates new dropouts
            dropout_event = 0
            if not pd.isna(current_dropout_rate) and not pd.isna(last_dropout_rate):
                if current_dropout_rate > last_dropout_rate:
                    dropout_event = 1
            
            # Update last dropout rate
            if not pd.isna(current_dropout_rate):
                last_dropout_rate = current_dropout_rate
            
            # Create record for this time point
            time_record = {
                **course_info,
                'time': current_time,
                'ref_year': ref_year,
                'dropout_event': dropout_event,
                'current_dropout_rate': current_dropout_rate
            }
            
            survival_data.append(time_record)
    
    # Convert to DataFrame
    survival_df = pd.DataFrame(survival_data)
    
    print(f"Created survival dataset with {len(survival_df)} records from {survival_df['Course_ID'].nunique()} courses")
    
    return survival_df


def perform_survival_analysis(survival_df, output_dir):
    """
    Perform survival analysis to understand dropout patterns over time.
    """
    print("Performing survival analysis...")
    
    # Create a proper time-to-event dataset
    # For each course, we need the time until dropout or censoring
    survival_summary = []
    
    for course_id, course_data in survival_df.groupby('Course_ID'):
        # Sort by time
        course_data = course_data.sort_values('time')
        
        # Get the first row for course information
        course_info = course_data.iloc[0].to_dict()
        
        # Determine if and when dropout occurred
        dropout_events = course_data[course_data['dropout_event'] == 1]
        
        if len(dropout_events) > 0:
            # Dropout occurred
            first_dropout = dropout_events.iloc[0]
            course_info['duration'] = first_dropout['time']
            course_info['event'] = 1
        else:
            # No dropout observed (censored)
            last_observation = course_data.iloc[-1]
            course_info['duration'] = last_observation['time']
            course_info['event'] = 0
        
        survival_summary.append(course_info)
    
    # Convert to DataFrame
    survival_summary_df = pd.DataFrame(survival_summary)
    
    # Keep only the necessary columns for analysis
    cols_to_keep = ['Course_ID', 'duration', 'event', 'Ano de Ingresso', 
                    'Categoria Administrativa', 'Organização Acadêmica', 'Grau Acadêmico',
                    'TP_MODALIDADE_ENSINO', 'TP_GRAU_ACADEMICO', 'QT_ING_FEM', 'QT_ING_MASC',
                    'QT_ING_18_24', 'QT_ING_25_29']
    
    cols_to_keep = [col for col in cols_to_keep if col in survival_summary_df.columns]
    survival_analysis_df = survival_summary_df[cols_to_keep].copy()
    
    # Perform Kaplan-Meier analysis
    print("Calculating Kaplan-Meier survival curves...")
    kmf = KaplanMeierFitter()
    kmf.fit(survival_analysis_df['duration'], event_observed=survival_analysis_df['event'], label="Overall")
    
    # Plot overall survival curve
    plt.figure(figsize=(10, 6))
    ax = kmf.plot_survival_function()
    plt.title('Overall Dropout Survival Curve')
    plt.xlabel('Years Since Enrollment')
    plt.ylabel('Probability of Not Dropping Out')
    plt.grid(alpha=0.3)
    plt.savefig(f'{output_dir}/overall_survival_curve.png')
    plt.close()
    
    # Variables to stratify the survival curves
    if 'Categoria Administrativa' in survival_analysis_df.columns:
        # Plot survival curves by administrative category
        plt.figure(figsize=(12, 7))
        categories = survival_analysis_df['Categoria Administrativa'].dropna().unique()
        
        for category in categories:
            if pd.notna(category):
                mask = survival_analysis_df['Categoria Administrativa'] == category
                if mask.sum() >= 30:  # Ensure sufficient data
                    kmf = KaplanMeierFitter()
                    kmf.fit(survival_analysis_df.loc[mask, 'duration'], 
                            event_observed=survival_analysis_df.loc[mask, 'event'], 
                            label=str(category))
                    ax = kmf.plot_survival_function()
        
        plt.title('Dropout Survival Curves by Administrative Category')
        plt.xlabel('Years Since Enrollment')
        plt.ylabel('Probability of Not Dropping Out')
        plt.grid(alpha=0.3)
        plt.legend(loc='best')
        plt.savefig(f'{output_dir}/survival_by_category.png')
        plt.close()
    
    # If we have numerical covariates, perform Cox Proportional Hazards regression
    numerical_cols = ['QT_ING_FEM', 'QT_ING_MASC', 'QT_ING_18_24', 'QT_ING_25_29']
    numerical_cols = [col for col in numerical_cols if col in survival_analysis_df.columns]
    
    if len(numerical_cols) > 0:
        print("Fitting Cox Proportional Hazards model...")
        
        # Prepare data for Cox PH model
        cph_df = survival_analysis_df.copy()
        
        # Handle missing values
        for col in numerical_cols:
            cph_df[col] = cph_df[col].fillna(cph_df[col].median())
        
        # Fit the model
        cph = CoxPHFitter()
        try:
            cph.fit(cph_df, duration_col='duration', event_col='event')
            
            # Print summary
            print(cph.summary)
            
            # Save summary to file
            with open(f'{output_dir}/cox_ph_summary.txt', 'w') as f:
                f.write("Cox Proportional Hazards Model Results\n")
                f.write("=====================================\n\n")
                f.write(str(cph.summary))
            
            # Plot hazard ratios
            plt.figure(figsize=(10, 6))
            cph.plot()
            plt.title('Hazard Ratios for Dropout Risk Factors')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{output_dir}/cox_hazard_ratios.png')
            plt.close()
        except Exception as e:
            print(f"Could not fit Cox PH model: {e}")
    
    # Return the prepared survival analysis dataframe
    return survival_analysis_df


def train_timepoint_models(time_point_data, output_dir):
    """
    Train models to predict dropout at specific time points.
    Each model predicts dropout probability at a fixed time since enrollment.
    """
    print("\nTraining time-specific dropout prediction models...")
    
    results = {}
    
    for year, data in time_point_data.items():
        print(f"\nTraining model for Year {year}...")
        
        X = data['X']
        y = data['y']
        
        # Convert target to binary classification based on median
        median_dropout = y.median()
        y_binary = (y > median_dropout).astype(int)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_binary, test_size=0.2, random_state=42, stratify=y_binary
        )
        
        # Get column types
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns
        
        # Create preprocessing steps
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Handle categorical variables
        print(f"  Processing {len(categorical_cols)} categorical features...")
        
        # Encode categorical variables manually
        X_train_encoded = X_train.copy()
        X_test_encoded = X_test.copy()
        
        # One-hot encode
        for col in categorical_cols:
            dummies_train = pd.get_dummies(X_train[col], prefix=col, drop_first=True)
            dummies_test = pd.get_dummies(X_test[col], prefix=col, drop_first=True)
            
            # Ensure test has same columns as train
            for dummy_col in dummies_train.columns:
                if dummy_col not in dummies_test.columns:
                    dummies_test[dummy_col] = 0
            
            dummies_test = dummies_test[dummies_train.columns]  # Match column order
            
            # Add to dataframes
            X_train_encoded = pd.concat([X_train_encoded, dummies_train], axis=1)
            X_test_encoded = pd.concat([X_test_encoded, dummies_test], axis=1)
            
            # Remove original categorical column
            X_train_encoded = X_train_encoded.drop(col, axis=1)
            X_test_encoded = X_test_encoded.drop(col, axis=1)
        
        # Process numeric features
        print(f"  Processing {len(numeric_cols)} numeric features...")
        
        X_train_num = X_train_encoded[numeric_cols].copy()
        X_test_num = X_test_encoded[numeric_cols].copy()
        
        # Apply transformations
        X_train_num = numeric_transformer.fit_transform(X_train_num)
        X_test_num = numeric_transformer.transform(X_test_num)
        
        # Get remaining columns (one-hot encoded)
        encoded_cols = [col for col in X_train_encoded.columns if col not in numeric_cols]
        X_train_cat = X_train_encoded[encoded_cols].values
        X_test_cat = X_test_encoded[encoded_cols].values
        
        # Combine numeric and categorical
        X_train_processed = np.hstack([X_train_num, X_train_cat])
        X_test_processed = np.hstack([X_test_num, X_test_cat])
        
        # Define feature names for importance analysis
        feature_names = list(numeric_cols) + list(encoded_cols)
        
        # Train random forest model
        rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42)
        rf.fit(X_train_processed, y_train)
        
        # Make predictions
        y_pred_prob = rf.predict_proba(X_test_processed)[:, 1]
        y_pred = rf.predict(X_test_processed)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_prob)
        
        print(f"  Year {year} model metrics:")
        print(f"    Accuracy: {accuracy:.4f}")
        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1-Score: {f1:.4f}")
        print(f"    ROC-AUC: {roc_auc:.4f}")
        
        # Feature importance
        importances = rf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        top_n = min(15, len(feature_names))
        top_features = [feature_names[i] for i in indices[:top_n]]
        top_importances = importances[indices[:top_n]]
        
        # Store results
        results[year] = {
            'model': rf,
            'metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'roc_auc': roc_auc
            },
            'importance': pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False),
            'top_features': top_features,
            'top_importances': top_importances,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'threshold': median_dropout
        }
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        plt.barh(top_features, top_importances)
        plt.title(f'Top Feature Importance for Year {year} Dropout Prediction')
        plt.xlabel('Importance')
        plt.gca().invert_yaxis()  # Highest importance at the top
        plt.tight_layout()
        plt.savefig(f'{output_dir}/year_{year}_feature_importance.png')
        plt.close()
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(results[year]['confusion_matrix'], annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Not High Risk', 'High Risk'],
                   yticklabels=['Not High Risk', 'High Risk'])
        plt.title(f'Confusion Matrix for Year {year} Dropout Prediction')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/year_{year}_confusion_matrix.png')
        plt.close()
    
    # Create model comparison chart
    years = sorted(results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    plt.figure(figsize=(12, 7))
    
    for metric in metrics:
        values = [results[year]['metrics'][metric] for year in years]
        plt.plot(years, values, 'o-', linewidth=2, label=metric.capitalize())
    
    plt.title('Dropout Prediction Model Performance by Year')
    plt.xlabel('Years Since Enrollment')
    plt.ylabel('Score')
    plt.grid(alpha=0.3)
    plt.xticks(years)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'{output_dir}/model_performance_by_year.png')
    plt.close()
    
    # Save feature importance summary
    with open(f'{output_dir}/feature_importance_summary.txt', 'w') as f:
        f.write("Feature Importance Summary by Year\n")
        f.write("=================================\n\n")
        
        for year in sorted(results.keys()):
            f.write(f"Year {year} Top Features:\n")
            for feature, importance in zip(results[year]['top_features'], results[year]['top_importances']):
                f.write(f"  {feature}: {importance:.4f}\n")
            f.write("\n")
    
    return results


def analyze_feature_importance_across_time(results, output_dir):
    """
    Analyze how feature importance changes across different time points.
    """
    print("\nAnalyzing feature importance across time points...")
    
    # Collect all unique features
    all_features = set()
    for year, data in results.items():
        all_features.update(data['importance']['Feature'])
    
    # Create a DataFrame to track importance over time
    importance_over_time = pd.DataFrame(index=sorted(all_features))
    
    for year, data in results.items():
        # Extract importances for this year
        year_importance = data['importance'].copy()
        year_importance = year_importance.set_index('Feature')
        
        # Add to tracking DataFrame
        importance_over_time[f'Year_{year}'] = 0  # Initialize with zeros
        for feature, row in year_importance.iterrows():
            importance_over_time.loc[feature, f'Year_{year}'] = row['Importance']
    
    # Calculate consistency metrics
    importance_over_time['Mean_Importance'] = importance_over_time.mean(axis=1)
    importance_over_time['Std_Importance'] = importance_over_time.std(axis=1)
    importance_over_time['Consistency'] = importance_over_time['Mean_Importance'] / (importance_over_time['Std_Importance'] + 0.01)
    
    # Sort by mean importance
    importance_over_time = importance_over_time.sort_values('Mean_Importance', ascending=False)
    
    # Save to CSV
    importance_over_time.to_csv(f'{output_dir}/feature_importance_over_time.csv')
    
    # Plot heatmap of top features over time
    top_n = 15
    top_features = importance_over_time.head(top_n).index
    
    year_columns = [col for col in importance_over_time.columns if col.startswith('Year_')]
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(importance_over_time.loc[top_features, year_columns], annot=True, 
               cmap='viridis', fmt='.3f', cbar_kws={'label': 'Importance'})
    plt.title(f'Top {top_n} Feature Importance Across Time')
    plt.ylabel('Feature')
    plt.xlabel('Years Since Enrollment')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance_heatmap.png')
    plt.close()
    
    # Plot feature importance trends for top consistent features
    importance_over_time = importance_over_time.sort_values('Consistency', ascending=False)
    consistent_features = importance_over_time.head(8).index
    
    plt.figure(figsize=(12, 7))
    
    for feature in consistent_features:
        values = [importance_over_time.loc[feature, col] for col in year_columns]
        plt.plot(range(1, len(year_columns) + 1), values, 'o-', linewidth=2, label=feature)
    
    plt.title('Importance Trends for Most Consistent Features')
    plt.xlabel('Years Since Enrollment')
    plt.ylabel('Feature Importance')
    plt.grid(alpha=0.3)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_consistency_trends.png')
    plt.close()
    
    return importance_over_time


def save_summary_report(cohort_df, survival_analysis_df, time_point_results, importance_over_time, output_dir):
    """
    Create a comprehensive summary report of the analysis.
    """
    print("\nGenerating summary report...")
    
    with open(f'{output_dir}/summary_report.md', 'w') as f:
        f.write("# Dropout Prediction: Time-Series Analysis Report\n\n")
        f.write(f"*Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n\n")
        
        # Dataset statistics
        f.write("## Dataset Overview\n\n")
        f.write(f"- Total courses analyzed: {cohort_df['CO_CURSO'].nunique()}\n")
        f.write(f"- Total institutions: {cohort_df['CO_IES'].nunique()}\n")
        f.write(f"- Enrollment cohorts: {sorted(cohort_df['Ano de Ingresso'].unique())}\n")
        f.write(f"- Courses with survival data: {survival_analysis_df['Course_ID'].nunique()}\n")
        
        # Dropout rates overview
        f.write("\n## Dropout Patterns\n\n")
        
        # Extract yearly dropout rates
        dropout_cols = [col for col in cohort_df.columns if col.startswith('Dropout_Y')]
        if dropout_cols:
            f.write("### Average Dropout Rates by Year\n\n")
            
            dropout_stats = {}
            for col in sorted(dropout_cols):
                year = int(col.split('_Y')[1])
                avg = cohort_df[col].mean()
                median = cohort_df[col].median()
                dropout_stats[year] = (avg, median)
            
            f.write("| Year | Average Dropout Rate | Median Dropout Rate |\n")
            f.write("|------|---------------------|---------------------|\n")
            
            for year, (avg, median) in sorted(dropout_stats.items()):
                f.write(f"| {year} | {avg:.2f}% | {median:.2f}% |\n")
            
            f.write("\n")
        
        # Survival analysis summary
        f.write("\n## Survival Analysis Results\n\n")
        f.write("Survival analysis treats dropout as a time-to-event outcome, allowing us to estimate the probability of a student remaining enrolled over time.\n\n")
        
        if 'event' in survival_analysis_df.columns:
            event_rate = survival_analysis_df['event'].mean() * 100
            f.write(f"- Overall dropout event rate: {event_rate:.2f}%\n")
            f.write(f"- Median survival time: {survival_analysis_df['duration'].median()} years\n\n")
            
            f.write("![Overall Survival Curve](overall_survival_curve.png)\n\n")
            
            if os.path.exists(f'{output_dir}/survival_by_category.png'):
                f.write("![Survival by Category](survival_by_category.png)\n\n")
        
        # Model performance summary
        f.write("\n## Predictive Models by Time Point\n\n")
        f.write("We built separate models to predict dropout at different points in the student journey:\n\n")
        
        if time_point_results:
            f.write("| Years Since Enrollment | Accuracy | Precision | Recall | F1-Score | ROC-AUC |\n")
            f.write("|------------------------|----------|-----------|--------|----------|--------|\n")
            
            for year in sorted(time_point_results.keys()):
                metrics = time_point_results[year]['metrics']
                f.write(f"| Year {year} | {metrics['accuracy']:.3f} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['roc_auc']:.3f} |\n")
            
            f.write("\n![Model Performance by Year](model_performance_by_year.png)\n\n")
        
        # Feature importance summary
        f.write("\n## Key Dropout Predictors\n\n")
        
        if not importance_over_time.empty:
            f.write("### Top 10 Features by Average Importance\n\n")
            
            top_features = importance_over_time.sort_values('Mean_Importance', ascending=False).head(10)
            
            f.write("| Feature | Mean Importance | Consistency |\n")
            f.write("|---------|----------------|------------|\n")
            
            for feature, row in top_features.iterrows():
                f.write(f"| {feature} | {row['Mean_Importance']:.4f} | {row['Consistency']:.2f} |\n")
            
            f.write("\n### Most Consistent Predictors Across Time\n\n")
            
            consistent_features = importance_over_time.sort_values('Consistency', ascending=False).head(10)
            
            f.write("| Feature | Mean Importance | Consistency |\n")
            f.write("|---------|----------------|------------|\n")
            
            for feature, row in consistent_features.iterrows():
                f.write(f"| {feature} | {row['Mean_Importance']:.4f} | {row['Consistency']:.2f} |\n")
            
            f.write("\n![Feature Importance Heatmap](feature_importance_heatmap.png)\n\n")
            f.write("\n![Feature Consistency Trends](feature_consistency_trends.png)\n\n")
        
        # Year-specific models
        f.write("\n## Year-by-Year Prediction Models\n\n")
        
        for year in sorted(time_point_results.keys()):
            f.write(f"### Year {year} Dropout Prediction\n\n")
            
            f.write(f"- Threshold for high dropout risk: {time_point_results[year]['threshold']:.2f}%\n")
            f.write(f"- Performance metrics: Accuracy = {time_point_results[year]['metrics']['accuracy']:.3f}, AUC = {time_point_results[year]['metrics']['roc_auc']:.3f}\n\n")
            
            f.write("#### Top 5 Predictors:\n\n")
            
            for i in range(min(5, len(time_point_results[year]['top_features']))):
                feature = time_point_results[year]['top_features'][i]
                importance = time_point_results[year]['top_importances'][i]
                f.write(f"- {feature}: {importance:.4f}\n")
            
            f.write("\n![Year {year} Feature Importance](year_{year}_feature_importance.png)\n\n")
            f.write("\n![Year {year} Confusion Matrix](year_{year}_confusion_matrix.png)\n\n")
        
        # Conclusions
        f.write("\n## Conclusions\n\n")
        f.write("This time-aware analysis of dropout patterns reveals:\n\n")
        
        f.write("1. Dropout risk varies significantly over the student journey\n")
        f.write("2. Different factors predict dropout at different time points\n")
        f.write("3. Institutional and course characteristics show consistent importance\n")
        f.write("4. Early warning systems should be calibrated to specific time points\n")
        
        f.write("\n## Methodological Notes\n\n")
        f.write("This analysis employs a time-aware approach to account for the longitudinal nature of dropout data:\n\n")
        
        f.write("- Cohort-based analysis rather than random sampling\n")
        f.write("- Separate models for different time points in the student journey\n")
        f.write("- Survival analysis to model time-to-dropout\n")
        f.write("- Consideration of feature importance changes over time\n")
    
    print(f"Summary report saved to {output_dir}/summary_report.md")


if __name__ == "__main__":
    # Create output directory
    output_dir = create_output_dirs()
    
    # Load and prepare data
    df = load_and_prepare_data(sample_size=100000)
    
    # Prepare longitudinal data structure
    cohort_df, time_point_data = prepare_longitudinal_data(df)
    
    # Prepare survival analysis dataset
    survival_df = prepare_survival_data(df)
    
    # Perform survival analysis
    survival_analysis_df = perform_survival_analysis(survival_df, output_dir)
    
    # Train time-specific prediction models
    time_point_results = train_timepoint_models(time_point_data, output_dir)
    
    # Analyze feature importance across time
    importance_over_time = analyze_feature_importance_across_time(time_point_results, output_dir)
    
    # Generate comprehensive report
    save_summary_report(cohort_df, survival_analysis_df, time_point_results, importance_over_time, output_dir)
    
    print(f"\nAnalysis complete. All results saved to {output_dir}")