# %%
# =============================================================================
# PROYECTO KAGGLE PROFESIONAL: PREDICCIÓN DE FELICIDAD (CIS 3462)
# =============================================================================
# Técnicas implementadas:
#   1. Clasificación supervisada (RandomForest, SVM, LogisticRegression, GradientBoosting)
#   2. Reglas de Asociación (Apriori via mlxtend)
#   3. AutoML - Comparación automática de modelos (LazyPredict)
# =============================================================================

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Reglas de asociación
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# AutoML – comparación automática de modelos
from lazypredict.Supervised import LazyClassifier


# %%
# =============================================================================
# 1. CARGA Y LIMPIEZA DE DATOS
# =============================================================================
df = pd.read_csv('3462_num.csv', sep=';', low_memory=False)

# Variables de salud/bienestar y objetivo (V1 = nivel de felicidad)
variables_salud = ['V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V52']
target = 'V1'

# Convertimos a numérico; valores no parseables pasan a NaN
for col in variables_salud + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminamos NaN y códigos de no-respuesta CIS (8 y 9)
df_model = df.dropna(subset=variables_salud + [target])
for col in variables_salud + [target]:
    df_model = df_model[df_model[col] <= 7]

# One-Hot Encoding para V52 (variable binaria: 1=Sí, 2=No)
df_model = pd.get_dummies(df_model, columns=['V52'], prefix='V52')
cols_v52 = [c for c in df_model.columns if c.startswith('V52_')]
variables_salud = [c for v in variables_salud for c in (cols_v52 if v == 'V52' else [v])]


# %%
# =============================================================================
# 2. ETIQUETADO: CLASIFICACIÓN EN 3 CLASES DE FELICIDAD
# =============================================================================
def categorizar_multiclass(val):
    if val <= 2: return 'Feliz'
    if val <= 4: return 'Medianamente_Feliz'
    return 'Infeliz'

def categorizar_binary(val):
    if val <= 4: return 'Feliz'
    return 'Infeliz'

# Elegir clasificación: 'multiclass' (3 clases) o 'binary' (2 clases)
clasificacion = 'multiclass'

df_model = df_model.copy()
if clasificacion == 'multiclass':
    df_model['Clase_Felicidad'] = df_model[target].apply(categorizar_multiclass)
else:
    df_model['Clase_Felicidad'] = df_model[target].apply(categorizar_binary)

print("Distribución de clases:")
print(df_model['Clase_Felicidad'].value_counts())


# %%
# =============================================================================
# 3. CLASIFICACIÓN SUPERVISADA (sklearn)
# =============================================================================
X = df_model[variables_salud]
y = df_model['Clase_Felicidad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

modelos = {
    'RandomForest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
    'SVM': SVC(kernel='rbf', C=1.0, random_state=42),
    'LogisticRegression': LogisticRegression(random_state=42, max_iter=1000),
    'GradientBoosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

resultados_sklearn = {}

for nombre, modelo in modelos.items():
    if nombre == 'SVM':
        modelo.fit(X_train_scaled, y_train)
        y_pred = modelo.predict(X_test_scaled)
    else:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    resultados_sklearn[nombre] = acc

    print(f"\n--- RESULTADO CON {nombre} ({clasificacion}) ---")
    print(f"Precisión: {acc:.2%}")
    print(f"Margen de Error: {1 - acc:.2%}")
    print("\nDesglose de aciertos por grupo:")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
    print("Matriz de confusión:")
    print(pd.DataFrame(cm, index=np.unique(y_test), columns=np.unique(y_test)))

    if hasattr(modelo, 'feature_importances_'):
        importancias = pd.Series(modelo.feature_importances_, index=variables_salud).sort_values(ascending=False)
        print(f"\nRanking de variables más influyentes ({nombre}):")
        print(importancias)
    elif nombre == 'LogisticRegression':
        importancias = pd.Series(np.abs(modelo.coef_[0]), index=variables_salud).sort_values(ascending=False)
        print(f"\nRanking de coeficientes absolutos ({nombre}):")
        print(importancias)

print("\n=== RESUMEN CLASIFICACIÓN SUPERVISADA ===")
for nombre, acc in sorted(resultados_sklearn.items(), key=lambda x: x[1], reverse=True):
    print(f"  {nombre:25s}: {acc:.2%}")


# %%
# =============================================================================
# 4. REGLAS DE ASOCIACIÓN (Apriori - mlxtend)
# =============================================================================
# Las reglas de asociación descubren patrones del tipo:
#   "Si una persona valora su salud física como ALTA y duerme BIEN
#    → es probable que sea FELIZ"
#
# Para aplicar Apriori necesitamos variables binarias/categóricas.
# Discretizamos cada variable numérica en tres niveles:
#   Bajo (1-2), Medio (3-5), Alto (6-7)
# =============================================================================

print("\n" + "="*60)
print("REGLAS DE ASOCIACIÓN (Apriori)")
print("="*60)

# Variables originales sin one-hot (usamos la base limpia antes del encoding)
variables_ar = ['V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V52', 'V1']
df_ar = df.copy()
for col in variables_ar:
    df_ar[col] = pd.to_numeric(df_ar[col], errors='coerce')
df_ar = df_ar.dropna(subset=variables_ar)
for col in variables_ar:
    df_ar = df_ar[df_ar[col] <= 7]

etiquetas_var = {
    'V1':  'Felicidad',
    'V32': 'SaludFisica',
    'V33': 'SaludMental',
    'V34': 'Suenyo',
    'V35': 'Energia',
    'V36': 'ActividadFisica',
    'V37': 'Dieta',
    'V52': 'TrabajoPagado',
}

def discretizar(val, nombre):
    if nombre == 'V52':
        return f"{etiquetas_var[nombre]}=Sí" if val == 1 else f"{etiquetas_var[nombre]}=No"
    if val <= 2:
        return f"{etiquetas_var[nombre]}=Alto"
    if val <= 4:
        return f"{etiquetas_var[nombre]}=Medio"
    return f"{etiquetas_var[nombre]}=Bajo"

# Construimos transacciones: cada fila es una lista de ítems discretizados
transacciones = []
for _, fila in df_ar[variables_ar].iterrows():
    items = [discretizar(fila[col], col) for col in variables_ar]
    transacciones.append(items)

# Codificación binaria para mlxtend
te = TransactionEncoder()
te_array = te.fit_transform(transacciones)
df_te = pd.DataFrame(te_array, columns=te.columns_)

# Minería de ítems frecuentes (soporte mínimo = 15%)
frecuentes = apriori(df_te, min_support=0.15, use_colnames=True)
frecuentes['longitud'] = frecuentes['itemsets'].apply(len)

print(f"\nÍtems frecuentes encontrados: {len(frecuentes)}")
print(frecuentes[frecuentes['longitud'] == 1].sort_values('support', ascending=False).head(10).to_string(index=False))

# Generación de reglas (confianza mínima = 60%)
reglas = association_rules(frecuentes, metric='confidence', min_threshold=0.60, num_itemsets=len(frecuentes))
reglas = reglas.sort_values('lift', ascending=False)

print(f"\nReglas de asociación generadas: {len(reglas)}")

# Filtramos reglas cuyo consecuente sea la felicidad
reglas_felicidad = reglas[
    reglas['consequents'].apply(lambda x: any('Felicidad' in str(item) for item in x))
]
reglas_felicidad = reglas_felicidad.sort_values('lift', ascending=False)

print(f"\nReglas que predicen nivel de FELICIDAD (consecuente = Felicidad): {len(reglas_felicidad)}")
if not reglas_felicidad.empty:
    print("\nTop 10 reglas por Lift:")
    cols_mostrar = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    print(reglas_felicidad[cols_mostrar].head(10).to_string(index=False))
else:
    print("\nTop 10 reglas generales por Lift:")
    cols_mostrar = ['antecedents', 'consequents', 'support', 'confidence', 'lift']
    print(reglas[cols_mostrar].head(10).to_string(index=False))


# %%
# =============================================================================
# 5. AUTOML CON LAZYPREDICT – COMPARACIÓN AUTOMÁTICA DE MODELOS
# =============================================================================
# LazyPredict automatiza la comparación de decenas de clasificadores sklearn
# sin necesidad de entrenarlos manualmente uno a uno.
# =============================================================================

print("\n" + "="*60)
print("AUTOML CON LAZYPREDICT")
print("="*60)

clf_lazy = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
modelos_lazy, pred_lazy = clf_lazy.fit(X_train, X_test, y_train, y_test)

print("\n=== RANKING AUTOMÁTICO DE MODELOS (LazyPredict) ===")
print(modelos_lazy.sort_values('Accuracy', ascending=False).head(15).to_string())

mejor_lazy = modelos_lazy.sort_values('Accuracy', ascending=False).index[0]
mejor_acc_lazy = modelos_lazy.sort_values('Accuracy', ascending=False).iloc[0]['Accuracy']
print(f"\n✓ Mejor modelo automático: {mejor_lazy}  (Accuracy = {mejor_acc_lazy:.4f})")


# %%
# =============================================================================
# 6. RESUMEN EJECUTIVO
# =============================================================================
print("\n" + "="*60)
print("RESUMEN EJECUTIVO DEL PROYECTO")
print("="*60)
print(f"\nDataset: Encuesta CIS 3462 — {len(df_model)} registros válidos")
print(f"Variables predictoras: {variables_salud}")
print(f"Variable objetivo: Clase_Felicidad ({clasificacion})")

print("\n[Clasificación Supervisada - sklearn]")
mejor_sklearn = max(resultados_sklearn, key=resultados_sklearn.get)
print(f"  Mejor modelo manual: {mejor_sklearn} ({resultados_sklearn[mejor_sklearn]:.2%})")

print("\n[Reglas de Asociación - Apriori]")
print(f"  Ítems frecuentes (soporte ≥ 15%): {len(frecuentes)}")
print(f"  Reglas generadas (confianza ≥ 60%): {len(reglas)}")
print(f"  Reglas que predicen felicidad: {len(reglas_felicidad)}")

print("\n[AutoML - LazyPredict]")
print(f"  Mejor modelo automático: {mejor_lazy}  (Accuracy = {mejor_acc_lazy:.4f})")