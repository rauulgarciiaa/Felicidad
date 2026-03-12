import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Carga de datos
df = pd.read_csv('3462_num.csv', sep=';', low_memory=False)

# Definimos variables según tu reto
variables_salud = ['V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V52']
target = 'V1'

# 2. LIMPIEZA PROFUNDA (Evita el error de la imagen)
# Convertimos todo a numérico. Los espacios "" o textos raros se vuelven NaN automáticamente
for col in variables_salud + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. FILTRADO DE VALORES DE ENCUESTA
# En el CIS, 8 y 9 suelen ser "No sabe/No contesta". Los eliminamos.
# También eliminamos las filas que quedaron con NaN tras el paso anterior.
df_model = df.dropna(subset=variables_salud + [target])

for col in variables_salud + [target]:
    df_model = df_model[df_model[col] <= 7] # Filtramos códigos de no respuesta

# 3.5 NORMALIZAR V52 A ESCALA 1-7
# V52 es binaria (1=Sí, 2=No), la convertimos a escala comparable (1-7)
df_model['V52'] = df_model['V52'].map({1: 7, 2: 1})  # Sí→7 (positivo), No→1 (negativo)

# 4. AGRUPACIÓN EN 3 CLASES o BINARIA
def categorizar_multiclass(val):
    if val <= 2: return 'Feliz'
    if val <= 4: return 'Medianamente Feliz'
    return 'Infeliz'

# Siempre multiclass para one-vs-rest
df_model = df_model.copy()
df_model['Clase_Felicidad'] = df_model[target].apply(categorizar_multiclass)

# 5. MODELO
X = df_model[variables_salud]
y = df_model['Clase_Felicidad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenamiento one-vs-rest para cada clase
clases = y.unique()
for clase in clases:
    y_train_bin = (y_train == clase).astype(int)
    y_test_bin = (y_test == clase).astype(int)
    modelo = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    modelo.fit(X_train, y_train_bin)
    y_pred = modelo.predict(X_test)
    print(f"\n--- MODELO PARA '{clase}' (vs resto) ---")
    print(f"Precisión: {accuracy_score(y_test_bin, y_pred):.2%}")
    print(f"Margen de Error: {1 - accuracy_score(y_test_bin, y_pred):.2%}")
    print("\nDesglose de aciertos:")
    print(classification_report(y_test_bin, y_pred, target_names=[f"No {clase}", clase]))
    importancias = pd.Series(modelo.feature_importances_, index=variables_salud).sort_values(ascending=False)
    print(f"\nRanking de preguntas que más influyen en la predicción de '{clase}':")
    print(importancias)