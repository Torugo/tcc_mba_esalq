import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp
from statsmodels.robust.robust_linear_model import RLM
import os

# Configurações para visualizações de alta qualidade
plt.rcParams['figure.dpi'] = 600
plt.rcParams['savefig.dpi'] = 600
sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'sans-serif'

# Carregando o dataset limpo
print("Carregando dados limpos...")
df_clean = pd.read_parquet('data/processed/trajetoria_academica_limpo.parquet')
print(f"Dataset carregado: {df_clean.shape[0]:,} registros e {df_clean.shape[1]} colunas")

# Após carregar os dados
print("\nVerificando valores ausentes:")
missing = df_clean.isnull().sum()[df_clean.isnull().sum() > 0]
if len(missing) > 0:
    print(missing)
else:
    print("Não há valores ausentes.")
    
print("\nVerificando tipos de dados:")
print(df_clean.dtypes.value_counts())

colunas_numericas = ['Taxa de Desistência Acumulada - TDA', 'Taxa de Desistência Anual - TADA',
                    'Tempo no Curso', 'Prazo de Integralização em Anos']

for col in colunas_numericas:
    if col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# Removendo linhas com valores ausentes nas colunas críticas
df_clean = df_clean.dropna(subset=['Taxa de Desistência Acumulada - TDA'])

# Verificando codificações das variáveis categóricas
for col in ['Categoria Administrativa Descrição', 'Modalidade de Ensino Descrição', 
            'Grau Acadêmico Descrição', 'Região Geográfica']:
    print(f"\nValores em '{col}':")
    print(df_clean[col].value_counts())

# 1. Análise Estatística por Categoria Administrativa (usando Kruskal-Wallis)
print("\n1. Análise Estatística por Categoria Administrativa (não-paramétrica)")

# Criando grupos para Kruskal-Wallis
grupos_admin = {}
for categoria in df_clean['Categoria Administrativa Descrição'].unique():
    if pd.notna(categoria):  # Ignorando valores NaN
        valores = df_clean[df_clean['Categoria Administrativa Descrição'] == categoria]['Taxa de Desistência Acumulada - TDA'].dropna()
        if len(valores) > 0:
            grupos_admin[categoria] = valores.values

# Realizando Kruskal-Wallis
anova_groups = list(grupos_admin.values())
h_stat, p_value = stats.kruskal(*anova_groups)
print(f"Resultados do teste Kruskal-Wallis:")
print(f"Estatística H: {h_stat:.4f}")
print(f"Valor p: {p_value:.8f}")
print(f"Pelo menos uma mediana é significativamente diferente (5%): {'Sim' if p_value < 0.05 else 'Não'}")

# Medianas dos grupos
for categoria, valores in grupos_admin.items():
    print(f"Mediana para {categoria}: {np.median(valores):.2f}%")

# Teste post-hoc Dunn
if p_value < 0.05:
    print("\nResultados do teste post-hoc de Dunn:")
    
    # Preparando dados para o teste de Dunn
    all_values = []
    all_labels = []
    for categoria, valores in grupos_admin.items():
        all_values.extend(valores)
        all_labels.extend([categoria] * len(valores))
    
    # Convertendo para DataFrame para usar scikit_posthocs
    dunn_df = pd.DataFrame({'grupo': all_labels, 'valor': all_values})
    
    # Realizando teste de Dunn
    try:
        dunn_results = sp.posthoc_dunn(dunn_df, val_col='valor', group_col='grupo', p_adjust='bonferroni')
        print(dunn_results)
    except Exception as e:
        print(f"Erro ao realizar teste de Dunn: {e}")
        print("Realizando comparações par a par manualmente:")
        # Alternativa: comparações par a par usando Mann-Whitney U
        categorias = list(grupos_admin.keys())
        for i in range(len(categorias)):
            for j in range(i+1, len(categorias)):
                cat1, cat2 = categorias[i], categorias[j]
                u_stat, p_val = stats.mannwhitneyu(grupos_admin[cat1], grupos_admin[cat2])
                print(f"{cat1} vs {cat2}: U={u_stat:.1f}, p={p_val:.8f} {'*' if p_val < 0.05 else ''}")

# Visualização das medianas com intervalos de confiança
plt.figure(figsize=(12, 8))
admin_stats = df_clean.groupby('Categoria Administrativa Descrição')['Taxa de Desistência Acumulada - TDA'].agg(
    ['median', 'count']).reset_index()

# Calculando intervalo de confiança para a mediana usando bootstrap
boot_ci = {}
for categoria in admin_stats['Categoria Administrativa Descrição']:
    data = df_clean[df_clean['Categoria Administrativa Descrição'] == categoria]['Taxa de Desistência Acumulada - TDA'].dropna()
    if len(data) > 30:  # Bootstrapping confiável apenas para amostras suficientemente grandes
        boot_samples = np.random.choice(data, size=(1000, len(data)), replace=True)
        boot_medians = np.median(boot_samples, axis=1)
        ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])
        boot_ci[categoria] = (ci_low, ci_high)
    else:
        # Para amostras pequenas, usamos um método aproximado
        boot_ci[categoria] = (data.median() - 1.96 * data.std() / np.sqrt(len(data)),
                             data.median() + 1.96 * data.std() / np.sqrt(len(data)))

# Adicionando intervalos de confiança ao DataFrame
admin_stats['ci_low'] = admin_stats['Categoria Administrativa Descrição'].map(lambda x: boot_ci[x][0])
admin_stats['ci_high'] = admin_stats['Categoria Administrativa Descrição'].map(lambda x: boot_ci[x][1])
admin_stats['error_low'] = admin_stats['median'] - admin_stats['ci_low']
admin_stats['error_high'] = admin_stats['ci_high'] - admin_stats['median']
admin_stats = admin_stats.sort_values('median', ascending=False)

# Plotagem com barras de erro assimétricas
plt.bar(admin_stats['Categoria Administrativa Descrição'], admin_stats['median'], color='skyblue', width=0.6)
plt.errorbar(x=admin_stats['Categoria Administrativa Descrição'], y=admin_stats['median'],
             yerr=[admin_stats['error_low'], admin_stats['error_high']], fmt='none', ecolor='black', capsize=5)

plt.title('Taxa de Desistência por Categoria Administrativa\ncom Intervalos de Confiança de 95% (Mediana)', fontsize=14)
plt.xlabel('Categoria Administrativa', fontsize=12)
plt.ylabel('Taxa de Desistência Acumulada - Mediana (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionando valores nas barras
for i, v in enumerate(admin_stats['median']):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('figuras/11-kruskal_categoria_administrativa.png', bbox_inches='tight')
plt.close()

# 2. Análise Estatística por Modalidade de Ensino (usando Mann-Whitney U)
print("\n2. Análise Estatística por Modalidade de Ensino (não-paramétrica)")

# Verificando valores únicos
print("Valores únicos em 'Modalidade de Ensino Descrição':")
print(df_clean['Modalidade de Ensino Descrição'].value_counts())

# Criando grupos para teste Mann-Whitney somente se ambas modalidades existirem
modalidades_disponiveis = df_clean['Modalidade de Ensino Descrição'].dropna().unique()

if len(modalidades_disponiveis) >= 2 and 'Presencial' in modalidades_disponiveis and 'Ensino a Distância' in modalidades_disponiveis:
    grupo_presencial = df_clean[df_clean['Modalidade de Ensino Descrição'] == 'Presencial']['Taxa de Desistência Acumulada - TDA'].dropna().values
    grupo_ead = df_clean[df_clean['Modalidade de Ensino Descrição'] == 'Ensino a Distância']['Taxa de Desistência Acumulada - TDA'].dropna().values
    
    # Verificando se ambos grupos têm tamanho suficiente
    if len(grupo_presencial) > 10 and len(grupo_ead) > 10:  # Tamanho mínimo para Mann-Whitney
        # Realizando teste Mann-Whitney U
        u_stat, p_value = stats.mannwhitneyu(grupo_presencial, grupo_ead, alternative='two-sided')
        print(f"Resultados do Teste Mann-Whitney U:")
        print(f"Estatística U: {u_stat:.4f}")
        print(f"Valor p: {p_value:.8f}")
        print(f"Diferença significativa a 5%: {'Sim' if p_value < 0.05 else 'Não'}")
        
        # Medianas dos grupos
        print(f"Mediana Presencial: {np.median(grupo_presencial):.2f}% (n={len(grupo_presencial):,})")
        print(f"Mediana EaD: {np.median(grupo_ead):.2f}% (n={len(grupo_ead):,})")
        print(f"Diferença: {np.median(grupo_ead) - np.median(grupo_presencial):.2f} pontos percentuais")
    else:
        print("Aviso: Um ou ambos os grupos têm tamanho insuficiente para o teste Mann-Whitney U.")
        if len(grupo_presencial) > 0 and len(grupo_ead) > 0:
            print(f"Mediana Presencial: {np.median(grupo_presencial):.2f}% (n={len(grupo_presencial):,})")
            print(f"Mediana EaD: {np.median(grupo_ead):.2f}% (n={len(grupo_ead):,})")
else:
    print("Aviso: Não há dados suficientes para ambas as modalidades de ensino para realizar o teste.")
    for modalidade in modalidades_disponiveis:
        grupo = df_clean[df_clean['Modalidade de Ensino Descrição'] == modalidade]['Taxa de Desistência Acumulada - TDA'].dropna()
        if len(grupo) > 0:
            print(f"Mediana {modalidade}: {grupo.median():.2f}% (n={len(grupo):,})")

# 3. Análise Estatística por Grau Acadêmico (usando Kruskal-Wallis)
print("\n3. Análise Estatística por Grau Acadêmico (não-paramétrica)")

# Criando grupos para Kruskal-Wallis
grupos_grau = {}
for grau in df_clean['Grau Acadêmico Descrição'].unique():
    if pd.notna(grau):  # Ignorando valores NaN
        valores = df_clean[df_clean['Grau Acadêmico Descrição'] == grau]['Taxa de Desistência Acumulada - TDA'].dropna()
        if len(valores) > 0:
            grupos_grau[grau] = valores.values

# Realizando Kruskal-Wallis
anova_groups = list(grupos_grau.values())
h_stat, p_value = stats.kruskal(*anova_groups)
print(f"Resultados do teste Kruskal-Wallis:")
print(f"Estatística H: {h_stat:.4f}")
print(f"Valor p: {p_value:.8f}")
print(f"Pelo menos uma mediana é significativamente diferente (5%): {'Sim' if p_value < 0.05 else 'Não'}")

# Medianas dos grupos
for grau, valores in grupos_grau.items():
    print(f"Mediana para {grau}: {np.median(valores):.2f}%")

# Teste post-hoc Dunn
if p_value < 0.05:
    print("\nResultados do teste post-hoc de Dunn:")
    
    # Preparando dados para o teste de Dunn
    all_values = []
    all_labels = []
    for grau, valores in grupos_grau.items():
        all_values.extend(valores)
        all_labels.extend([grau] * len(valores))
    
    # Convertendo para DataFrame para usar scikit_posthocs
    dunn_df = pd.DataFrame({'grupo': all_labels, 'valor': all_values})
    
    # Realizando teste de Dunn
    try:
        dunn_results = sp.posthoc_dunn(dunn_df, val_col='valor', group_col='grupo', p_adjust='bonferroni')
        print(dunn_results)
    except Exception as e:
        print(f"Erro ao realizar teste de Dunn: {e}")
        print("Realizando comparações par a par manualmente:")
        # Alternativa: comparações par a par usando Mann-Whitney U
        graus = list(grupos_grau.keys())
        for i in range(len(graus)):
            for j in range(i+1, len(graus)):
                grau1, grau2 = graus[i], graus[j]
                u_stat, p_val = stats.mannwhitneyu(grupos_grau[grau1], grupos_grau[grau2])
                print(f"{grau1} vs {grau2}: U={u_stat:.1f}, p={p_val:.8f} {'*' if p_val < 0.05 else ''}")

# Visualização das medianas com intervalos de confiança
plt.figure(figsize=(10, 6))
grau_stats = df_clean.groupby('Grau Acadêmico Descrição')['Taxa de Desistência Acumulada - TDA'].agg(
    ['median', 'count']).reset_index()

# Calculando intervalo de confiança para a mediana usando bootstrap
boot_ci = {}
for grau in grau_stats['Grau Acadêmico Descrição']:
    data = df_clean[df_clean['Grau Acadêmico Descrição'] == grau]['Taxa de Desistência Acumulada - TDA'].dropna()
    if len(data) > 30:
        boot_samples = np.random.choice(data, size=(1000, len(data)), replace=True)
        boot_medians = np.median(boot_samples, axis=1)
        ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])
        boot_ci[grau] = (ci_low, ci_high)
    else:
        boot_ci[grau] = (data.median() - 1.96 * data.std() / np.sqrt(len(data)),
                         data.median() + 1.96 * data.std() / np.sqrt(len(data)))

# Adicionando intervalos de confiança ao DataFrame
grau_stats['ci_low'] = grau_stats['Grau Acadêmico Descrição'].map(lambda x: boot_ci[x][0])
grau_stats['ci_high'] = grau_stats['Grau Acadêmico Descrição'].map(lambda x: boot_ci[x][1])
grau_stats['error_low'] = grau_stats['median'] - grau_stats['ci_low']
grau_stats['error_high'] = grau_stats['ci_high'] - grau_stats['median']
grau_stats = grau_stats.sort_values('median', ascending=False)

# Plotagem com barras de erro assimétricas
plt.bar(grau_stats['Grau Acadêmico Descrição'], grau_stats['median'], color='salmon', width=0.6)
plt.errorbar(x=grau_stats['Grau Acadêmico Descrição'], y=grau_stats['median'],
             yerr=[grau_stats['error_low'], grau_stats['error_high']], fmt='none', ecolor='black', capsize=5)

plt.title('Taxa de Desistência por Grau Acadêmico\ncom Intervalos de Confiança de 95% (Mediana)', fontsize=14)
plt.xlabel('Grau Acadêmico', fontsize=12)
plt.ylabel('Taxa de Desistência Acumulada Mediana (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionando valores nas barras
for i, v in enumerate(grau_stats['median']):
    plt.text(i, v + 2, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('figuras/13-kruskal_grau_academico.png', bbox_inches='tight')
plt.close()

# 4. Análise Estatística por Região Geográfica (usando Kruskal-Wallis)
print("\n4. Análise Estatística por Região Geográfica (não-paramétrica)")

# Criando grupos para Kruskal-Wallis
grupos_regiao = {}
for regiao in df_clean['Região Geográfica'].unique():
    if pd.notna(regiao):  # Ignorando valores NaN
        valores = df_clean[df_clean['Região Geográfica'] == regiao]['Taxa de Desistência Acumulada - TDA'].dropna()
        if len(valores) > 0:
            grupos_regiao[regiao] = valores.values

# Realizando Kruskal-Wallis
anova_groups = list(grupos_regiao.values())
h_stat, p_value = stats.kruskal(*anova_groups)
print(f"Resultados do teste Kruskal-Wallis:")
print(f"Estatística H: {h_stat:.4f}")
print(f"Valor p: {p_value:.8f}")
print(f"Pelo menos uma mediana é significativamente diferente (5%): {'Sim' if p_value < 0.05 else 'Não'}")

# Medianas dos grupos
for regiao, valores in grupos_regiao.items():
    print(f"Mediana para {regiao}: {np.median(valores):.2f}%")

# Teste post-hoc Dunn
if p_value < 0.05:
    print("\nResultados do teste post-hoc de Dunn:")
    
    # Preparando dados para o teste de Dunn
    all_values = []
    all_labels = []
    for regiao, valores in grupos_regiao.items():
        all_values.extend(valores)
        all_labels.extend([regiao] * len(valores))
    
    # Convertendo para DataFrame para usar scikit_posthocs
    dunn_df = pd.DataFrame({'grupo': all_labels, 'valor': all_values})
    
    # Realizando teste de Dunn
    try:
        dunn_results = sp.posthoc_dunn(dunn_df, val_col='valor', group_col='grupo', p_adjust='bonferroni')
        print(dunn_results)
    except Exception as e:
        print(f"Erro ao realizar teste de Dunn: {e}")
        print("Realizando comparações par a par manualmente:")
        # Alternativa: comparações par a par usando Mann-Whitney U
        regioes = list(grupos_regiao.keys())
        for i in range(len(regioes)):
            for j in range(i+1, len(regioes)):
                regiao1, regiao2 = regioes[i], regioes[j]
                u_stat, p_val = stats.mannwhitneyu(grupos_regiao[regiao1], grupos_regiao[regiao2])
                print(f"{regiao1} vs {regiao2}: U={u_stat:.1f}, p={p_val:.8f} {'*' if p_val < 0.05 else ''}")

# Visualização das medianas com intervalos de confiança
plt.figure(figsize=(12, 6))
regiao_stats = df_clean.groupby('Região Geográfica')['Taxa de Desistência Acumulada - TDA'].agg(
    ['median', 'count']).reset_index()

# Calculando intervalo de confiança para a mediana usando bootstrap
boot_ci = {}
for regiao in regiao_stats['Região Geográfica']:
    data = df_clean[df_clean['Região Geográfica'] == regiao]['Taxa de Desistência Acumulada - TDA'].dropna()
    if len(data) > 30:
        boot_samples = np.random.choice(data, size=(1000, len(data)), replace=True)
        boot_medians = np.median(boot_samples, axis=1)
        ci_low, ci_high = np.percentile(boot_medians, [2.5, 97.5])
        boot_ci[regiao] = (ci_low, ci_high)
    else:
        boot_ci[regiao] = (data.median() - 1.96 * data.std() / np.sqrt(len(data)),
                           data.median() + 1.96 * data.std() / np.sqrt(len(data)))

# Adicionando intervalos de confiança ao DataFrame
regiao_stats['ci_low'] = regiao_stats['Região Geográfica'].map(lambda x: boot_ci[x][0])
regiao_stats['ci_high'] = regiao_stats['Região Geográfica'].map(lambda x: boot_ci[x][1])
regiao_stats['error_low'] = regiao_stats['median'] - regiao_stats['ci_low']
regiao_stats['error_high'] = regiao_stats['ci_high'] - regiao_stats['median']
regiao_stats = regiao_stats.sort_values('median', ascending=False)

# Plotagem com barras de erro assimétricas
plt.bar(regiao_stats['Região Geográfica'], regiao_stats['median'], color='lightblue', width=0.6)
plt.errorbar(x=regiao_stats['Região Geográfica'], y=regiao_stats['median'],
             yerr=[regiao_stats['error_low'], regiao_stats['error_high']], fmt='none', ecolor='black', capsize=5)

plt.title('Taxa de Desistência por Região Geográfica\ncom Intervalos de Confiança de 95% (Mediana)', fontsize=14)
plt.xlabel('Região Geográfica', fontsize=12)
plt.ylabel('Taxa de Desistência Acumulada Mediana (%)', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Adicionando valores nas barras
for i, v in enumerate(regiao_stats['median']):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center')

plt.tight_layout()
plt.savefig('figuras/14-kruskal_regiao_geografica.png', bbox_inches='tight')
plt.close()

# 5. Análise de Regressão: Efeito do Tempo no Curso na Taxa de Desistência (usando regressão robusta)
print("\n5. Análise de Regressão Robusta: Efeito do Tempo no Curso na Taxa de Desistência")

# Preparando os dados para regressão robusta
X = sm.add_constant(df_clean['Tempo no Curso'])
y = df_clean['Taxa de Desistência Acumulada - TDA']

# Ajustando o modelo robusto
modelo_robusto = RLM(y, X, M=sm.robust.norms.HuberT()).fit()

# Imprimindo os resultados
print(modelo_robusto.summary())

# Calculando valores ajustados
df_clean['valores_ajustados_robustos'] = modelo_robusto.predict(X)

# Visualizando a regressão robusta
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Tempo no Curso', y='Taxa de Desistência Acumulada - TDA', 
               data=df_clean, alpha=0.1)
sns.lineplot(x='Tempo no Curso', y='valores_ajustados_robustos', 
            data=df_clean, color='red', linewidth=2)
plt.title('Regressão Robusta: Tempo no Curso x Taxa de Desistência', fontsize=14)
plt.xlabel('Anos no Curso', fontsize=12)
plt.ylabel('Taxa de Desistência Acumulada (%)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('figuras/15-regressao_robusta_tempo_desistencia.png', bbox_inches='tight')
plt.close()

# 6. Análise de Regressão Múltipla (usando regressão robusta)
print("\n6. Análise de Regressão Múltipla Robusta")

# Criando um DataFrame completamente novo apenas com as variáveis numéricas
df_reg_new = pd.DataFrame()

# Adicionando variáveis contínuas
df_reg_new['Tempo_no_Curso'] = df_clean['Tempo no Curso'].astype(float)
df_reg_new['Prazo_Integralizacao'] = df_clean['Prazo de Integralização em Anos'].astype(float)

# Adicionando variáveis categóricas manualmente como numéricas (0 ou 1)
# Categoria Administrativa (usando Pública Federal como referência)
df_reg_new['Cat_Privada_Fins_Lucrativos'] = (df_clean['Categoria Administrativa Descrição'] == 'Privada com fins lucrativos').astype(float)
df_reg_new['Cat_Privada_Sem_Fins_Lucrativos'] = (df_clean['Categoria Administrativa Descrição'] == 'Privada sem fins lucrativos').astype(float)
df_reg_new['Cat_Publica_Estadual'] = (df_clean['Categoria Administrativa Descrição'] == 'Pública Estadual').astype(float)
df_reg_new['Cat_Publica_Municipal'] = (df_clean['Categoria Administrativa Descrição'] == 'Pública Municipal').astype(float)

# Grau Acadêmico (usando Bacharelado como referência)
df_reg_new['Grau_Licenciatura'] = (df_clean['Grau Acadêmico Descrição'] == 'Licenciatura').astype(float)
df_reg_new['Grau_Tecnologico'] = (df_clean['Grau Acadêmico Descrição'] == 'Tecnológico').astype(float)

# Modalidade de Ensino (usando Presencial como referência)
df_reg_new['Modalidade_EaD'] = (df_clean['Modalidade de Ensino Descrição'] == 'Ensino a Distância').astype(float)

# Região (usando Sudeste como referência)
df_reg_new['Regiao_Norte'] = (df_clean['Região Geográfica'] == 'Norte').astype(float)
df_reg_new['Regiao_Nordeste'] = (df_clean['Região Geográfica'] == 'Nordeste').astype(float)
df_reg_new['Regiao_Sul'] = (df_clean['Região Geográfica'] == 'Sul').astype(float)
df_reg_new['Regiao_Centro_Oeste'] = (df_clean['Região Geográfica'] == 'Centro-Oeste').astype(float)

# Variável dependente
df_reg_new['Taxa_Desistencia'] = df_clean['Taxa de Desistência Acumulada - TDA'].astype(float)

# Remover linhas com valores NaN
df_reg_new = df_reg_new.dropna()

# Verificar os tipos de dados
print(f"Tipos de dados no novo DataFrame:")
print(df_reg_new.dtypes)
print(f"Dimensões: {df_reg_new.shape}")

# Preparar X e y
X = df_reg_new.drop('Taxa_Desistencia', axis=1)
y = df_reg_new['Taxa_Desistencia']

# Adicionar constante
X = sm.add_constant(X)

# Executar a regressão robusta
try:
    modelo_robusto = RLM(y, X, M=sm.robust.norms.HuberT()).fit()
    print(modelo_robusto.summary())
    
    # Extrair e classificar coeficientes
    coefs = pd.DataFrame({
        "Variável": X.columns[1:],
        "Coeficiente": modelo_robusto.params[1:],
        "Erro Padrão": modelo_robusto.bse[1:],
        "Valor p": modelo_robusto.pvalues[1:]
    })
    
    coefs = coefs.sort_values("Coeficiente", ascending=False)
    print("\nVariáveis com maior impacto positivo na taxa de desistência:")
    print(coefs.head(5))
    
    # Visualizando os principais coeficientes
    top_coefs = coefs.head(10)
    plt.figure(figsize=(12, 8))
    plt.barh(top_coefs['Variável'], top_coefs['Coeficiente'], color='blue')
    plt.title('Top 10 Fatores com Maior Impacto na Taxa de Desistência (Regressão Robusta)', fontsize=14)
    plt.xlabel('Coeficiente de Regressão', fontsize=12)
    plt.ylabel('Variável', fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('figuras/16-regressao_multipla_robusta_top_coeficientes.png', bbox_inches='tight')
    plt.close()
    
except Exception as e:
    print(f"Erro na regressão múltipla robusta: {e}")
    print("Não foi possível completar a análise de regressão múltipla robusta.")