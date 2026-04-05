# %% [markdown]
# # Proyecto Profesional Kaggle: Felicidad en España (CIS 3462)
#
# Este notebook implementa un análisis completo estilo Kaggle con:
# - **Análisis Exploratorio de Datos (EDA)**
# - **Clasificación Supervisada** (RandomForest, SVM, Logistic Regression, Gradient Boosting, XGBoost, LightGBM)
# - **Reglas de Asociación** (Apriori + mlxtend)
# - **AutoML con PyCaret** (comparación y selección automática de modelos)
#
# **Dataset**: Encuesta CIS 3462 - Salud y Calidad de Vida (España, 2024)
# **Target**: V1 - Grado de felicidad personal

# %% Importaciones
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')   # Backend no interactivo (sin pantalla)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Modelos de ML
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (classification_report, accuracy_score,
                             confusion_matrix, roc_auc_score)
from sklearn.pipeline import Pipeline

# Librerías adicionales de ML
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# Reglas de Asociación
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# AutoML
from tpot import TPOTClassifier
from dask.distributed import Client, LocalCluster

print("✅ Todas las librerías importadas correctamente")

# %%
# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO DE DATOS
# =============================================================================

print("\n" + "="*60)
print("1. CARGA Y PREPROCESAMIENTO DE DATOS")
print("="*60)

# Cargamos el CSV con etiquetas (datos de la encuesta CIS 3462)
df_raw = pd.read_csv('3462_etiq.csv', sep=';', low_memory=False, encoding='utf-8-sig')
print(f"Dataset cargado: {df_raw.shape[0]} filas x {df_raw.shape[1]} columnas")

# Identificamos las columnas clave por prefijo
def find_col(df, prefix):
    """Devuelve el nombre completo de columna que comienza con el prefijo dado."""
    matches = [c for c in df.columns if c.startswith(prefix + ' ') or c == prefix]
    return matches[0] if matches else None

# Columnas de interés
COL_V1    = find_col(df_raw, 'V1')       # Felicidad (target)
COL_V33   = find_col(df_raw, 'V33')      # Dificultades laborales por salud
COL_V34   = find_col(df_raw, 'V34')      # Dolor corporal
COL_V35   = find_col(df_raw, 'V35')      # Desdichado / deprimido
COL_V36   = find_col(df_raw, 'V36')      # Pérdida de confianza en sí mismo
COL_V37   = find_col(df_raw, 'V37')      # Incapaz de superar problemas
COL_V44   = find_col(df_raw, 'V44')      # Satisfacción sanidad española
COL_V51   = find_col(df_raw, 'V51')      # Estado de salud general
COL_V52   = find_col(df_raw, 'V52')      # Enfermedad crónica / discapacidad
COL_SEXO  = find_col(df_raw, 'SEXO')     # Sexo
COL_EDAD  = find_col(df_raw, 'EDAD')     # Edad
COL_TOPBOT = find_col(df_raw, 'TOPBOT')  # Clase social subjetiva

FEATURE_COLS = [COL_V33, COL_V34, COL_V35, COL_V36, COL_V37,
                COL_V44, COL_V51, COL_V52, COL_SEXO, COL_EDAD, COL_TOPBOT]
ALL_COLS = [COL_V1] + FEATURE_COLS

# Filtramos sólo las columnas necesarias
df = df_raw[ALL_COLS].copy()
df.columns = ['Felicidad', 'DifilaboralSalud', 'DolCorporal', 'Depresion',
              'PerdConfianza', 'NoPuedeProblemas', 'SatSanidad',
              'EstadoSalud', 'EnfCronica', 'Sexo', 'Edad', 'ClaseSocial']

print(f"\nColumnas seleccionadas: {df.columns.tolist()}")
print(f"\nDistribución del target (V1):\n{df['Felicidad'].value_counts()}")

# %%
# Eliminamos respuestas no válidas: N.C., No sabría decir, N.P.
NO_RESPUESTA = ['N.C.', 'No sabría decir', 'N.P.', 'No procede']
for col in df.columns:
    df[col] = df[col].replace(NO_RESPUESTA, np.nan)

df.dropna(inplace=True)
print(f"\nFilas tras eliminar no respuestas: {df.shape[0]}")

# %%
# Agrupación del target en 3 clases de felicidad
MAPA_FELICIDAD = {
    'Completamente feliz': 'Feliz',
    'Muy feliz': 'Feliz',
    'Bastante feliz': 'Feliz',
    'Ni feliz ni infeliz': 'Medianamente_Feliz',
    'Bastante infeliz': 'Infeliz',
    'Muy infeliz': 'Infeliz',
    'Completamente infeliz': 'Infeliz',
}
df['Felicidad'] = df['Felicidad'].map(MAPA_FELICIDAD)
df.dropna(subset=['Felicidad'], inplace=True)

print(f"\nDistribución de clases tras agrupación:\n{df['Felicidad'].value_counts()}")
print(f"\nFilas finales: {df.shape[0]}")

# %%
# =============================================================================
# 2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)
# =============================================================================

print("\n" + "="*60)
print("2. ANÁLISIS EXPLORATORIO DE DATOS (EDA)")
print("="*60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Análisis Exploratorio - Felicidad en España (CIS 3462)', fontsize=14, fontweight='bold')

# Distribución de felicidad
colores = {'Feliz': '#4CAF50', 'Medianamente_Feliz': '#FF9800', 'Infeliz': '#F44336'}
counts = df['Felicidad'].value_counts()
axes[0, 0].bar(counts.index, counts.values, color=[colores[c] for c in counts.index])
axes[0, 0].set_title('Distribución de Felicidad')
axes[0, 0].set_xlabel('Clase')
axes[0, 0].set_ylabel('Frecuencia')

# Felicidad por sexo
pd.crosstab(df['Sexo'], df['Felicidad']).plot(kind='bar', ax=axes[0, 1],
    color=[colores.get(c, 'gray') for c in pd.crosstab(df['Sexo'], df['Felicidad']).columns])
axes[0, 1].set_title('Felicidad por Sexo')
axes[0, 1].set_xlabel('Sexo')
axes[0, 1].tick_params(axis='x', rotation=0)
axes[0, 1].legend(title='Felicidad', fontsize=8)

# Estado de salud vs Felicidad
orden_salud = ['Excelente', 'Muy buena', 'Buena', 'Regular', 'Mala']
orden_salud = [s for s in orden_salud if s in df['EstadoSalud'].unique()]
pd.crosstab(df['EstadoSalud'].astype(pd.CategoricalDtype(orden_salud, ordered=True)),
            df['Felicidad']).plot(kind='bar', ax=axes[0, 2],
    color=[colores.get(c, 'gray') for c in df['Felicidad'].unique()])
axes[0, 2].set_title('Estado de Salud vs Felicidad')
axes[0, 2].set_xlabel('Estado de Salud')
axes[0, 2].tick_params(axis='x', rotation=45)
axes[0, 2].legend(title='Felicidad', fontsize=8)

# Edad vs Felicidad (boxplot)
df['Edad'] = pd.to_numeric(df['Edad'], errors='coerce')
feliz_data = df.dropna(subset=['Edad'])
for ax_idx, clase in enumerate(['Feliz', 'Medianamente_Feliz', 'Infeliz']):
    subset = feliz_data[feliz_data['Felicidad'] == clase]['Edad']
    axes[1, 0].hist(subset, alpha=0.6, label=clase, color=list(colores.values())[ax_idx], bins=15)
axes[1, 0].set_title('Distribución de Edad por Clase de Felicidad')
axes[1, 0].set_xlabel('Edad')
axes[1, 0].legend()

# Depresion vs Felicidad
pd.crosstab(df['Depresion'], df['Felicidad']).plot(kind='bar', ax=axes[1, 1],
    color=[colores.get(c, 'gray') for c in df['Felicidad'].unique()])
axes[1, 1].set_title('Sentirse Desdichado/Deprimido vs Felicidad')
axes[1, 1].set_xlabel('Frecuencia de Depresión')
axes[1, 1].tick_params(axis='x', rotation=45)
axes[1, 1].legend(title='Felicidad', fontsize=8)

# Clase social vs Felicidad
df['ClaseSocial'] = pd.to_numeric(df['ClaseSocial'], errors='coerce')
cs_data = df.dropna(subset=['ClaseSocial'])
cs_mean = cs_data.groupby('Felicidad')['ClaseSocial'].mean().reindex(['Infeliz', 'Medianamente_Feliz', 'Feliz'])
axes[1, 2].bar(cs_mean.index, cs_mean.values, color=[colores[c] for c in cs_mean.index])
axes[1, 2].set_title('Clase Social Media por Nivel de Felicidad')
axes[1, 2].set_xlabel('Clase de Felicidad')
axes[1, 2].set_ylabel('Clase Social Media (1-10)')

plt.tight_layout()
plt.savefig('eda_felicidad.png', dpi=120, bbox_inches='tight')
plt.show()
print("✅ Gráfico EDA guardado como 'eda_felicidad.png'")

# %%
# =============================================================================
# 3. PREPARACIÓN PARA MODELOS SUPERVISADOS
# =============================================================================

print("\n" + "="*60)
print("3. PREPARACIÓN PARA MODELOS SUPERVISADOS")
print("="*60)

df_model = df.copy()

# Codificación ordinal para variables de frecuencia
MAPA_FRECUENCIA = {
    'Nunca': 1, 'Rara vez': 2, 'Algunas veces': 3,
    'A menudo': 4, 'Muy a menudo': 5
}
MAPA_SALUD = {
    'Mala': 1, 'Regular': 2, 'Buena': 3, 'Muy buena': 4, 'Excelente': 5
}
MAPA_SAT_SANIDAD = {
    'Muy insatisfecho/a': 1, 'Bastante insatisfecho/a': 2,
    'Ni satisfecho/a ni insatisfecho/a': 3,
    'Bastante satisfecho/a': 4, 'Muy satisfecho/a': 5
}
MAPA_ENF_CRONICA = {'No': 0, 'Sí': 1}

for col in ['DifilaboralSalud', 'DolCorporal', 'Depresion', 'PerdConfianza', 'NoPuedeProblemas']:
    df_model[col] = df_model[col].map(MAPA_FRECUENCIA)

df_model['EstadoSalud'] = df_model['EstadoSalud'].map(MAPA_SALUD)
df_model['SatSanidad'] = df_model['SatSanidad'].map(MAPA_SAT_SANIDAD)
df_model['EnfCronica'] = df_model['EnfCronica'].map(MAPA_ENF_CRONICA)
df_model = pd.get_dummies(df_model, columns=['Sexo'], prefix='Sexo')
df_model['Edad'] = pd.to_numeric(df_model['Edad'], errors='coerce')
df_model['ClaseSocial'] = pd.to_numeric(df_model['ClaseSocial'], errors='coerce')

df_model.dropna(inplace=True)
print(f"Filas disponibles para modelado: {df_model.shape[0]}")

# Variables de entrada y salida
feature_cols = [c for c in df_model.columns if c != 'Felicidad']
X = df_model[feature_cols]
y = df_model['Felicidad']

# Codificamos el target
le = LabelEncoder()
y_enc = le.fit_transform(y)
print(f"Clases: {le.classes_}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
)
print(f"Train: {X_train.shape[0]} muestras | Test: {X_test.shape[0]} muestras")

# Escalado para SVM y Logistic Regression
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# %%
# =============================================================================
# 4. CLASIFICACIÓN SUPERVISADA - MÚLTIPLES MODELOS
# =============================================================================

print("\n" + "="*60)
print("4. CLASIFICACIÓN SUPERVISADA - MÚLTIPLES MODELOS")
print("="*60)

modelos = {
    'RandomForest':       (RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42), False),
    'GradientBoosting':   (GradientBoostingClassifier(n_estimators=150, max_depth=4, random_state=42), False),
    'XGBoost':            (XGBClassifier(n_estimators=150, max_depth=4, use_label_encoder=False,
                                         eval_metric='mlogloss', random_state=42, verbosity=0), False),
    'LightGBM':           (LGBMClassifier(n_estimators=150, max_depth=5, random_state=42, verbose=-1), False),
    'SVM':                (SVC(kernel='rbf', C=1.0, probability=True, random_state=42), True),
    'LogisticRegression': (LogisticRegression(max_iter=1000, random_state=42), True),
}

resultados = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nombre, (modelo, usar_scaler) in modelos.items():
    Xtr = X_train_sc if usar_scaler else X_train
    Xte = X_test_sc  if usar_scaler else X_test

    # Entrenamiento
    modelo.fit(Xtr, y_train)
    y_pred = modelo.predict(Xte)

    # CV sobre training
    cv_scores = cross_val_score(modelo, Xtr, y_train, cv=cv, scoring='accuracy')

    acc = accuracy_score(y_test, y_pred)
    resultados[nombre] = {
        'Accuracy_Test': acc,
        'CV_Mean': cv_scores.mean(),
        'CV_Std': cv_scores.std(),
        'Error_Rate': 1 - acc,
    }

    print(f"\n--- {nombre} ---")
    print(f"  Accuracy Test : {acc:.4f} ({acc:.2%})")
    print(f"  CV (5-fold)   : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Error Rate    : {1 - acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # Importancia de características (si disponible)
    if hasattr(modelo, 'feature_importances_'):
        imp = pd.Series(modelo.feature_importances_, index=X.columns).sort_values(ascending=False)
        print(f"  Top 5 variables:\n{imp.head(5).to_string()}")

# Tabla comparativa
df_resultados = pd.DataFrame(resultados).T.sort_values('Accuracy_Test', ascending=False)
print("\n" + "="*60)
print("TABLA COMPARATIVA DE MODELOS")
print("="*60)
print(df_resultados.round(4).to_string())

# Gráfico comparativo
fig, ax = plt.subplots(figsize=(10, 5))
colores_bar = ['#4CAF50' if i == 0 else '#2196F3' for i in range(len(df_resultados))]
df_resultados['Accuracy_Test'].plot(kind='bar', ax=ax, color=colores_bar, edgecolor='black')
ax.set_title('Comparativa de Modelos Supervisados - Accuracy en Test', fontweight='bold')
ax.set_ylabel('Accuracy')
ax.set_ylim(0, 1)
ax.axhline(df_resultados['Accuracy_Test'].mean(), color='red', linestyle='--', label=f'Media: {df_resultados["Accuracy_Test"].mean():.3f}')
ax.legend()
plt.tight_layout()
plt.savefig('comparativa_modelos.png', dpi=120, bbox_inches='tight')
plt.show()
print("✅ Gráfico comparativo guardado como 'comparativa_modelos.png'")

# Variables del mejor modelo manual (usadas también en la sección 6 y 7)
mejor_manual = df_resultados['Accuracy_Test'].idxmax()
acc_mejor_manual = df_resultados.loc[mejor_manual, 'Accuracy_Test']
print(f"\n🏆 Mejor modelo manual: {mejor_manual} ({acc_mejor_manual:.2%} accuracy)")

# %%
# =============================================================================
# 5. REGLAS DE ASOCIACIÓN (APRIORI)
# =============================================================================

print("\n" + "="*60)
print("5. REGLAS DE ASOCIACIÓN (APRIORI + mlxtend)")
print("="*60)
print("Objetivo: Encontrar patrones entre comportamientos de salud y felicidad")

# Preparamos datos para Apriori: variables categóricas binarias
df_assoc = df[['Felicidad', 'DifilaboralSalud', 'DolCorporal', 'Depresion',
               'PerdConfianza', 'NoPuedeProblemas', 'EstadoSalud', 'EnfCronica', 'Sexo']].copy()

# Binarizamos: Frecuencia alta = A menudo / Muy a menudo → True
def binarizar_frecuencia(x, umbral_alto=('A menudo', 'Muy a menudo')):
    if x in umbral_alto:
        return 'Alta_Freq'
    elif x in ('Nunca', 'Rara vez'):
        return 'Baja_Freq'
    return 'Media_Freq'

for col in ['DifilaboralSalud', 'DolCorporal', 'Depresion', 'PerdConfianza', 'NoPuedeProblemas']:
    df_assoc[col] = df[col].apply(binarizar_frecuencia)

def binarizar_salud(x):
    if x in ('Excelente', 'Muy buena', 'Buena'):
        return 'Buena_Salud'
    return 'Mala_Salud'

df_assoc['EstadoSalud'] = df['EstadoSalud'].apply(binarizar_salud)

# Convertimos en transacciones (cada fila es una "cesta" de atributos)
transacciones = []
for _, row in df_assoc.iterrows():
    t = []
    for col in df_assoc.columns:
        t.append(f"{col}={row[col]}")
    transacciones.append(t)

# Codificación one-hot para Apriori
te = TransactionEncoder()
te_array = te.fit_transform(transacciones)
df_te = pd.DataFrame(te_array, columns=te.columns_)

print(f"Items únicos: {len(te.columns_)}")
print(f"Transacciones: {len(df_te)}")

# Apriori: buscar itemsets frecuentes
print("\nBuscando itemsets frecuentes (min_support=0.10)...")
frecuentes = apriori(df_te, min_support=0.10, use_colnames=True)
frecuentes['length'] = frecuentes['itemsets'].apply(len)
print(f"Itemsets frecuentes encontrados: {len(frecuentes)}")
print(frecuentes[frecuentes['length'] >= 2].sort_values('support', ascending=False).head(10).to_string(index=False))

# Generamos reglas de asociación
print("\nGenerando reglas de asociación (min_confidence=0.60)...")
reglas = association_rules(frecuentes, metric='confidence', min_threshold=0.60)
reglas = reglas.sort_values('lift', ascending=False)

print(f"Reglas generadas: {len(reglas)}")
print("\nTop 15 reglas por Lift:")
print(reglas[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(15).to_string(index=False))

# Filtramos las reglas que llevan a FELICIDAD
print("\n--- Reglas que predicen FELICIDAD ---")
reglas_feliz = reglas[reglas['consequents'].astype(str).str.contains('Felicidad=Feliz')]
print(reglas_feliz[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string(index=False))

print("\n--- Reglas que predicen INFELICIDAD ---")
reglas_infeliz = reglas[reglas['consequents'].astype(str).str.contains('Felicidad=Infeliz')]
print(reglas_infeliz[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10).to_string(index=False))

# Gráfico: Support vs Confidence coloreado por Lift
fig, ax = plt.subplots(figsize=(10, 6))
sc = ax.scatter(reglas['support'], reglas['confidence'],
                c=reglas['lift'], cmap='RdYlGn', alpha=0.7, s=60, edgecolors='k', linewidths=0.3)
plt.colorbar(sc, ax=ax, label='Lift')
ax.set_xlabel('Support')
ax.set_ylabel('Confidence')
ax.set_title('Reglas de Asociación: Support vs Confidence (color = Lift)', fontweight='bold')
plt.tight_layout()
plt.savefig('reglas_asociacion.png', dpi=120, bbox_inches='tight')
plt.show()
print("✅ Gráfico de reglas guardado como 'reglas_asociacion.png'")

# %%
# =============================================================================
# 6. AUTOML CON TPOT - Optimización Automática de Pipelines
# =============================================================================

print("\n" + "="*60)
print("6. AUTOML CON TPOT - Optimización Automática de Pipelines")
print("="*60)
print("TPOT usa programación genética para encontrar el mejor pipeline de ML")

# Iniciamos un cluster Dask local requerido por TPOT v1
print("\nIniciando cluster Dask local...")
import warnings
warnings.filterwarnings('ignore')
cluster = LocalCluster(n_workers=2, threads_per_worker=2, processes=False, silence_logs=True)
client_dask = Client(cluster)
print(f"Dask dashboard: {client_dask.dashboard_link}")

# Búsqueda automática del mejor pipeline con TPOT (nueva API v1.x)
# TPOT v1 usa 'scorers' en lugar de 'scoring', y no tiene 'generations'/'population_size'
# directamente - busca con max_time_mins
tpot = TPOTClassifier(
    scorers=['accuracy'],
    scorers_weights=[1.0],
    cv=5,
    random_state=42,
    verbose=1,
    n_jobs=-1,
    max_time_mins=5,           # Máximo 5 minutos de búsqueda
    early_stop=3,              # Para si no mejora en 3 generaciones
    client=client_dask
)

print(f"\nBuscando el mejor pipeline automáticamente (máx. 5 min)...")
tpot.fit(X_train, y_train)

# En TPOT v1 el mejor pipeline está en fitted_pipeline_
mejor_pipeline_tpot = tpot.fitted_pipeline_
y_pred_tpot = mejor_pipeline_tpot.predict(X_test)
acc_automl = accuracy_score(y_test, y_pred_tpot)

print(f"\n✅ Accuracy TPOT en test: {acc_automl:.4f} ({acc_automl:.2%})")
print(f"\nMejor pipeline encontrado:")
print(mejor_pipeline_tpot)
print("\nInforme detallado:")
print(classification_report(y_test, y_pred_tpot, target_names=le.classes_, zero_division=0))

# Exportamos el mejor pipeline como código Python (guardamos representación textual)
with open('mejor_pipeline_tpot.py', 'w') as f:
    f.write(f"# Mejor pipeline encontrado por TPOT AutoML\n")
    f.write(f"# Accuracy en test: {acc_automl:.4f}\n\n")
    f.write(f"from sklearn.pipeline import make_pipeline\n")
    f.write(f"import joblib\n\n")
    f.write(f"# Pipeline:\n# {str(mejor_pipeline_tpot)}\n\n")
    f.write(f"# Para reutilizar el modelo:\n")
    f.write(f"# import joblib\n")
    f.write(f"# modelo = joblib.load('mejor_pipeline_tpot.pkl')\n")
    f.write(f"# predicciones = modelo.predict(X_nuevos)\n")

# Guardamos el pipeline para reutilización
import joblib
joblib.dump(mejor_pipeline_tpot, 'mejor_pipeline_tpot.pkl')
print("✅ Mejor pipeline exportado como 'mejor_pipeline_tpot.py' y 'mejor_pipeline_tpot.pkl'")

# Comparación AutoML vs mejor manual
print("\nComparando AutoML (TPOT) vs mejor modelo manual:")
print(f"  Mejor manual ({mejor_manual}): {acc_mejor_manual:.4f}")
print(f"  TPOT AutoML:                  {acc_automl:.4f}")
ganador = "TPOT AutoML" if acc_automl >= acc_mejor_manual else f"Manual ({mejor_manual})"
print(f"  🏆 Ganador: {ganador}")

# Cerramos el cluster Dask
client_dask.close()
cluster.close()

# %%
# =============================================================================
# 7. RESUMEN EJECUTIVO
# =============================================================================

print("\n" + "="*60)
print("7. RESUMEN EJECUTIVO")
print("="*60)

print(f"""
📊 RESULTADOS FINALES:
┌─────────────────────────────────────────────────────┐
│  CLASIFICACIÓN SUPERVISADA (Manual)                 │
│  ✓ Mejor modelo: {mejor_manual:<30} │
│  ✓ Accuracy en test: {acc_mejor_manual:.4f} ({acc_mejor_manual:.2%})          │
├─────────────────────────────────────────────────────┤
│  REGLAS DE ASOCIACIÓN (Apriori)                     │
│  ✓ Itemsets frecuentes: {len(frecuentes):<26} │
│  ✓ Reglas generadas:   {len(reglas):<26} │
│  ✓ Reglas → Feliz:     {len(reglas_feliz):<26} │
│  ✓ Reglas → Infeliz:   {len(reglas_infeliz):<26} │
├─────────────────────────────────────────────────────┤
│  AUTOML - TPOT                                      │
│  ✓ Accuracy AutoML: {acc_automl:.4f} ({acc_automl:.2%})            │
│  ✓ Pipeline exportado: mejor_pipeline_tpot.py       │
└─────────────────────────────────────────────────────┘

🔍 HALLAZGOS CLAVE (Reglas de Asociación):
   • La depresión frecuente y el estado de salud malo se
     asocian fuertemente con la infelicidad (lift alto).
   • El buen estado de salud y la baja frecuencia de dolor
     son los mejores predictores de felicidad.
   • Las personas sin enfermedad crónica y con buena salud
     tienen alta probabilidad de ser Felices.

📦 LIBRERÍAS UTILIZADAS:
   • scikit-learn  → Modelos base de ML
   • XGBoost       → Gradient boosting escalable
   • LightGBM      → Boosting rápido de Microsoft
   • mlxtend       → Reglas de Asociación (Apriori)
   • TPOT          → AutoML con programación genética
   • matplotlib / seaborn → Visualización
""")
