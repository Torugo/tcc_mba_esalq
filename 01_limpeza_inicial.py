# data_cleaning.py
import pandas as pd
import numpy as np
import os

def clean_dataset():
    """
    Realiza a limpeza inicial do dataset:
    - Remove colunas com alta porcentagem de valores ausentes
    - Remove linhas com valores ausentes em variáveis importantes
    - Salva o dataset limpo para uso posterior
    """
    print("Iniciando limpeza dos dados...")
    
    # Carregar os dados
    df = pd.read_parquet('data/joined_data.parquet')
    
    # Verificar dimensões e tipos de dados
    print(f"Dimensões do DataFrame original: {df.shape}")
    print("\nTipos de dados:")
    print(df.dtypes.value_counts())
    
    # Analisar valores ausentes
    missing_data = df.isnull().sum().sort_values(ascending=False)
    missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
    missing = pd.concat([missing_data, missing_percent], axis=1, keys=['Missing Values', 'Percentage'])
    print("\nTop 15 colunas com valores ausentes:")
    print(missing[missing['Missing Values'] > 0].head(15))
    
    # Remover colunas com mais de 50% de valores ausentes
    high_missing = missing[missing['Percentage'] > 50].index.tolist()
    df_clean = df.drop(columns=high_missing)
    print(f"\nColunas removidas por terem mais de 50% de valores ausentes: {len(high_missing)}")
    print(f"Dimensões após remoção: {df_clean.shape}")
    
    # Remover linhas com valores ausentes nas variáveis principais
    main_vars = ['Taxa de Desistência Acumulada - TDA', 'Categoria Administrativa', 
                 'Grau Acadêmico', 'NO_REGIAO', 'Modalidade de Ensino']
    
    # Verificar se todas as variáveis principais existem no dataframe
    existing_main_vars = [var for var in main_vars if var in df_clean.columns]
    df_clean = df_clean.dropna(subset=existing_main_vars)
    print(f"Dimensões após remoção de linhas com valores ausentes nas variáveis principais: {df_clean.shape}")
    
    # Criar diretório para dados processados, se não existir
    if not os.path.exists('data/processed'):
        os.makedirs('data/processed')
    
    # Salvar o dataset limpo
    df_clean.to_parquet('data/processed/clean_data.parquet')
    print("Dataset limpo salvo em 'data/processed/clean_data.parquet'")
    
    return df_clean

if __name__ == "__main__":
    clean_dataset()