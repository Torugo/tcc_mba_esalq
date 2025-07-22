import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.robust.robust_linear_model import RLM

# === 1. Carregar dados =======================================================

df = pd.read_parquet("data/processed/trajetoria_academica_limpo.parquet")

# === 2. Garantir tipos numéricos =============================================

num_cols = [
    "Taxa de Desistência Acumulada - TDA",
    "Tempo no Curso",
    "Prazo de Integralização em Anos",
]
for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Mantém apenas linhas completas para as análises
df = df.dropna(subset=["Taxa de Desistência Acumulada - TDA", "Tempo no Curso"])

# === 3‑A. Regressão robusta simples (Tempo no Curso) =========================

X_simple = sm.add_constant(df["Tempo no Curso"])
y         = df["Taxa de Desistência Acumulada - TDA"]

modelo_simples = RLM(y, X_simple, M=sm.robust.norms.HuberT()).fit()

y_pred_simples = modelo_simples.predict(X_simple)
r2_simples = 1 - np.sum((y - y_pred_simples) ** 2) / np.sum((y - y.mean()) ** 2)

print("\n=== Regressão Robusta Simples ===")
print(modelo_simples.summary())
print(f"R² (calculado manualmente): {r2_simples:.4f}")

# === 3‑B. Regressão robusta múltipla =========================================

# Monta dataframe de regressão com dummies básicas
reg = pd.DataFrame({
    "Tempo_no_Curso"      : df["Tempo no Curso"].astype(float),
    "Prazo_Integralizacao": df["Prazo de Integralização em Anos"].astype(float),
    "Taxa_Desistencia"    : df["Taxa de Desistência Acumulada - TDA"].astype(float),

    # Dummies de exemplo (ajuste conforme necessário)
    "Modalidade_EaD"      : (df["Modalidade de Ensino Descrição"] == "Ensino a Distância").astype(float),
    "Grau_Licenciatura"   : (df["Grau Acadêmico Descrição"]       == "Licenciatura"       ).astype(float),
})

reg = reg.dropna()
X_mult = sm.add_constant(reg.drop(columns="Taxa_Desistencia"))
y_mult = reg["Taxa_Desistencia"]

modelo_mult = RLM(y_mult, X_mult, M=sm.robust.norms.HuberT()).fit()

y_pred_mult = modelo_mult.predict(X_mult)
r2_mult = 1 - np.sum((y_mult - y_pred_mult) ** 2) / np.sum((y_mult - y_mult.mean()) ** 2)

print("\n=== Regressão Robusta Múltipla ===")
print(modelo_mult.summary())
print(f"R² (calculado manualmente): {r2_mult:.4f}")
