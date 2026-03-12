# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# 1. Carga de datos
df = pd.read_csv('3462_num.csv', sep=';', low_memory=False)

# Definimos variables según tu reto
variables_salud = ['V24', 'V33', 'V34', 'V35', 'V36', 'V37', 'V48', 'V49', 'V50', 'V51', 'V52']
target = 'V1'

# 2. LIMPIEZA PROFUNDA (Evita el error de la imagen)
# Convertimos todo a numérico. Los espacios "" o textos raros se vuelven NaN automáticamente
for col in variables_salud + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. FILTRADO DE VALORES DE ENCUESTA
# En el CIS, 8 y 9 suelen ser "No sabe/No contesta". Los eliminamos.
# También eliminamos las filas que quedaron con NaN tras el paso anterior.
df_model = df[variables_salud + [target]].dropna()

for col in variables_salud + [target]:
    df_model = df_model[df_model[col] <= 7] # Filtramos códigos de no respuesta


# %%
# 3.5 ONE-HOT ENCODING PARA V52 (binaria: 1=Sí, 2=No)
df_model = pd.get_dummies(df_model, columns=['V52'], prefix='V52')
# Actualizamos la lista de variables reemplazando 'V52' por las columnas dummy generadas
cols_v52 = [c for c in df_model.columns if c.startswith('V52_')]
variables_salud = [c for v in variables_salud for c in (cols_v52 if v == 'V52' else [v])]


# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# 1. Carga de datos
df = pd.read_csv('3462_num.csv', sep=';', low_memory=False)

# Definimos variables según tu reto
variables_salud = ['V24', 'V33', 'V34', 'V35', 'V36', 'V37', 'V48', 'V49', 'V50', 'V51', 'V52']
target = 'V1'

# 2. LIMPIEZA PROFUNDA
for col in variables_salud + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_model = df[variables_salud + [target]].dropna()

for col in variables_salud + [target]:
    df_model = df_model[df_model[col] <= 7]

# %%
# 3. ONE-HOT ENCODING PARA V52
df_model = pd.get_dummies(df_model, columns=['V52'], prefix='V52')
cols_v52 = [c for c in df_model.columns if c.startswith('V52_')]
variables_salud = [c for v in variables_salud for c in (cols_v52 if v == 'V52' else [v])]

# %%
# 4. AGRUPACIÓN EN 3 CLASES
def categorizar(val):
    if val <= 2: return 'Feliz'
    if val <= 4: return 'Medianamente Feliz'
    return 'Infeliz'

df_model['Clase_Felicidad'] = df_model[target].apply(categorizar)

# 5. MODELOS SIMPLES MULTICLASE
X = df_model[variables_salud]
y = df_model['Clase_Felicidad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelos = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
}

for nombre, modelo in modelos.items():
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)
    
    print(f"\n--- RESULTADO CON {nombre.upper()} ---")
    print(f"Precisión: {accuracy_score(y_test, y_pred):.2%}")
    print(f"Margen de Error: {1 - accuracy_score(y_test, y_pred):.2%}")
    print("\nDesglose de aciertos por grupo:")
    print(classification_report(y_test, y_pred))

# 6. MEJORAS PARA SUBIR PRECISIÓN
print("\n" + "="*50)
print("SUGERENCIAS PARA MEJORAR LA PRECISIÓN:")
print("="*50)
print("1. Balanceo de clases: Usar class_weight='balanced' (ya aplicado en Logistic Regression)")
print("2. Ajuste de hiperparámetros: Probar diferentes valores de max_depth en Decision Tree o n_neighbors en KNN")
print("3. Selección de features: Usar solo las más importantes (V35, V37, V51 según importancia)")
print("4. Ensemble: Combinar predicciones de varios modelos")
print("5. Cross-validation: Usar validación cruzada para evaluación más robusta")