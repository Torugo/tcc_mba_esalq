# remove_redundant.py
import pandas as pd
import os

def remove_redundant_variables():
    """
    Identifica e remove variáveis redundantes do dataset:
    - Identifica pares/grupos de variáveis que representam a mesma informação
    - Mantém apenas uma variável de cada grupo (preferencialmente a mais legível)
    - Salva o dataset sem redundâncias para uso posterior
    """
    print("Iniciando remoção de variáveis redundantes...")
    
    # Verificar se o arquivo de dados limpos existe
    clean_file = 'data/processed/clean_data.parquet'
    if not os.path.exists(clean_file):
        print(f"Arquivo {clean_file} não encontrado. Execute data_cleaning.py primeiro.")
        return None
    
    # Carregar dados limpos
    df_clean = pd.read_parquet(clean_file)
    
    # Definir grupos de variáveis redundantes
    redundant_groups = [
        # Códigos/nomes de instituições
        ['Código da Instituição', 'CO_IES'],
        # Informações regionais
        ['NO_REGIAO', 'CO_REGIAO'],
        ['NO_UF', 'CO_UF', 'SG_UF'],
        ['NO_MUNICIPIO', 'CO_MUNICIPIO'],
        # Informações de cursos
        ['Nome do Curso de Graduação', 'NO_CURSO'],
        ['Código do Curso de Graduação', 'CO_CURSO'],
        # Classificação CINE
        ['Nome da área do Curso segundo a classificação CINE BRASIL', 'NO_CINE_ROTULO', 'CO_CINE_ROTULO'],
        ['Nome da Grande Área do Curso segundo a classificação CINE BRASIL', 'NO_CINE_AREA_GERAL'],
        # Diferentes contagens de ingressantes
        ['Quantidade de Ingressantes no Curso', 'QT_ING'],
        # Variáveis relacionadas à permanência/conclusão
        ['Quantidade de Permanência no Curso no ano de referência', 'QT_SIT_TRANCADA'],
        ['Quantidade de Concluintes no Curso no ano de referência', 'QT_CONC'],
        ['Quantidade de Desistência no Curso no ano de referência', 'QT_SIT_DESVINCULADO'], 
        ['Quantidade de Falecimentos no Curso no ano de referência', 'QT_SIT_FALECIDO'],
        # Source files
        ['source_file', 'source_file_1']
    ]
    
    # Para cada grupo, determinar a coluna a manter e as a remover
    cols_to_keep = [
        'Código da Instituição', 'NO_REGIAO', 'NO_UF', 'NO_MUNICIPIO', 
        'Nome do Curso de Graduação', 'Código do Curso de Graduação',
        'Nome da área do Curso segundo a classificação CINE BRASIL', 
        'Nome da Grande Área do Curso segundo a classificação CINE BRASIL',
        'Quantidade de Ingressantes no Curso', 
        'Quantidade de Permanência no Curso no ano de referência',
        'Quantidade de Concluintes no Curso no ano de referência', 
        'Quantidade de Desistência no Curso no ano de referência',
        'Quantidade de Falecimentos no Curso no ano de referência',
        'source_file'
    ]
    
    # Compilar lista de colunas a remover
    cols_to_drop = []
    for group in redundant_groups:
        # Identifica qual coluna manter neste grupo
        keep_col = None
        for col in group:
            if col in cols_to_keep:
                keep_col = col
                break
        
        if keep_col:
            # Adiciona as outras colunas do grupo à lista de remoção
            for col in group:
                if col != keep_col and col in df_clean.columns:
                    cols_to_drop.append(col)
    
    # Remover colunas redundantes
    df_no_redundant = df_clean.drop(columns=cols_to_drop, errors='ignore')
    
    print(f"Colunas redundantes removidas: {len(cols_to_drop)}")
    print(f"Dimensões após remoção de redundâncias: {df_no_redundant.shape}")
    
    # Salvar dataset sem redundâncias
    df_no_redundant.to_parquet('data/processed/no_redundant_data.parquet')
    print("Dataset sem redundâncias salvo em 'data/processed/no_redundant_data.parquet'")
    
    return df_no_redundant

if __name__ == "__main__":
    remove_redundant_variables()