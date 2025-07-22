# Análise da Taxa de Desistência no Ensino Superior Brasileiro
# Autor: [Seu Nome]
# Data: [Data]

# Importando bibliotecas necessárias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import os

# Configurações para visualizações de alta qualidade
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial']

# Criando diretório para salvar figuras se não existir
if not os.path.exists('figuras'):
    os.makedirs('figuras')

# Carregando os dados
print("Carregando dados de trajetória acadêmica...")
df = pd.read_parquet("data/indicadores_trajetoria/indicadores_trajetoria_educacao_superior_all.parquet")

# Exibindo informações básicas sobre o dataset
print(f"Dimensões do DataFrame: {df.shape}")
print("Tipos de dados:")
print(df.dtypes.value_counts())

# Verificando valores ausentes
missing_data = df.isnull().sum().sort_values(ascending=False)
missing_percent = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
missing = pd.concat([missing_data, missing_percent], axis=1, 
                   keys=['Valores Ausentes', 'Porcentagem'])
print("Top 10 colunas com valores ausentes:")
print(missing[missing['Valores Ausentes'] > 0].head(10))

# Estatísticas básicas das taxas de desistência
print("Estatísticas sobre taxas de desistência:")
dropout_stats = df[['Taxa de Desistência Acumulada - TDA', 'Taxa de Desistência Anual - TADA']].describe()
print(dropout_stats)

# 1. Limpeza e Preparação dos Dados

# 1.1 Verificando a consistência dos dados
print("\nVerificando consistência dos dados...")
print(f"Anos de ingresso únicos: {sorted(df['Ano de Ingresso'].unique())}")
print(f"Anos de referência únicos: {sorted(df['Ano de Referência'].unique())}")

# 1.2 Criando variáveis derivadas úteis para a análise
print("\nCriando variáveis derivadas...")

# Tempo no curso (ano de referência - ano de ingresso)
df['Tempo no Curso'] = df['Ano de Referência'] - df['Ano de Ingresso']

# Identificando categorias administrativas
cat_admin_map = {
    1.0: 'Pública Federal',
    2.0: 'Pública Estadual',
    3.0: 'Pública Municipal',
    4.0: 'Privada com fins lucrativos',
    5.0: 'Privada sem fins lucrativos',
    7.0: 'Especial'
}
df['Categoria Administrativa Descrição'] = df['Categoria Administrativa'].map(cat_admin_map)

# Identificando organizações acadêmicas
org_acad_map = {
    1.0: 'Universidade',
    2.0: 'Centro Universitário',
    3.0: 'Faculdade',
    4.0: 'Instituto Federal',
    5.0: 'Centro Federal de Educação Tecnológica'
}
df['Organização Acadêmica Descrição'] = df['Organização Acadêmica'].map(org_acad_map)

# Identificando graus acadêmicos
grau_acad_map = {
    1.0: 'Bacharelado',
    2.0: 'Licenciatura',
    3.0: 'Tecnológico'
}
df['Grau Acadêmico Descrição'] = df['Grau Acadêmico'].map(grau_acad_map)

# Identificando modalidades de ensino
mod_ensino_map = {
    1.0: 'Presencial',
    2.0: 'Ensino a Distância'
}
df['Modalidade de Ensino Descrição'] = df['Modalidade de Ensino'].map(mod_ensino_map)

# Identificando regiões geográficas
regiao_map = {
    1.0: 'Norte',
    2.0: 'Nordeste',
    3.0: 'Sudeste',
    4.0: 'Sul',
    5.0: 'Centro-Oeste'
}
df['Região Geográfica'] = df['Código da Região Geográfica do Curso'].map(regiao_map)

# 1.3 Removendo colunas redundantes ou com pouca relevância para análise
colunas_para_manter = [
    'Código da Instituição', 'Nome da Instituição', 
    'Categoria Administrativa', 'Categoria Administrativa Descrição',
    'Organização Acadêmica', 'Organização Acadêmica Descrição',
    'Nome do Curso de Graduação', 
    'Grau Acadêmico', 'Grau Acadêmico Descrição',
    'Modalidade de Ensino', 'Modalidade de Ensino Descrição',
    'Nome da área do Curso segundo a classificação CINE BRASIL',
    'Nome da Grande Área do Curso segundo a classificação CINE BRASIL',
    'Região Geográfica',
    'Ano de Ingresso', 'Ano de Referência', 'Tempo no Curso',
    'Prazo de Integralização em Anos',
    'Quantidade de Ingressantes no Curso',
    'Quantidade de Permanência no Curso no ano de referência',
    'Quantidade de Concluintes no Curso no ano de referência',
    'Quantidade de Desistência no Curso no ano de referência',
    'Taxa de Permanência - TAP',
    'Taxa de Conclusão Acumulada - TCA',
    'Taxa de Desistência Acumulada - TDA',
    'Taxa de Conclusão Anual - TCAN',
    'Taxa de Desistência Anual - TADA'
]

df_clean = df[colunas_para_manter].copy()
print(f"\nDataset após remoção de colunas redundantes: {df_clean.shape}")

# 1.4 Verificando valores anômalos ou incorretos nas taxas
# As taxas devem estar entre 0 e 100
print("\nVerificando valores anômalos nas taxas...")
taxas_colunas = ['Taxa de Permanência - TAP', 'Taxa de Conclusão Acumulada - TCA', 
                 'Taxa de Desistência Acumulada - TDA', 'Taxa de Conclusão Anual - TCAN', 
                 'Taxa de Desistência Anual - TADA']

for col in taxas_colunas:
    anomalias = ((df_clean[col] < 0) | (df_clean[col] > 100)).sum()
    print(f"Valores anômalos em {col}: {anomalias}")

# 1.5 Verificando a soma das taxas acumuladas (deve ser 100%)
df_clean['Soma das Taxas'] = df_clean['Taxa de Permanência - TAP'] + \
                           df_clean['Taxa de Conclusão Acumulada - TCA'] + \
                           df_clean['Taxa de Desistência Acumulada - TDA']

# Verificando se a soma é aproximadamente 100
tolerancia = 0.01  # Tolerância de 0.01% para erros de arredondamento
inconsistencias = ((df_clean['Soma das Taxas'] < 100-tolerancia) | 
                   (df_clean['Soma das Taxas'] > 100+tolerancia)).sum()
print(f"Registros com soma de taxas fora de 100% (±{tolerancia}%): {inconsistencias}")

# Removendo a coluna auxiliar após a verificação
df_clean.drop('Soma das Taxas', axis=1, inplace=True)

# Salvando o dataset limpo
df_clean.to_parquet('data/processed/trajetoria_academica_limpo.parquet')
print("\nDataset limpo salvo em 'data/processed/trajetoria_academica_limpo.parquet'")