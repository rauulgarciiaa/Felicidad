# %%
# SEGMENTO 1: IMPORTACIONES
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

# %%
# SEGMENTO 2: CARGA DE DATOS
df = pd.read_csv('3462_num.csv', sep=';', low_memory=False)

# Definir las variables independientes y la variable objetivo
variables_salud = ['V24', 'V33', 'V34', 'V35', 'V36', 'V37', 'V48', 'V49', 'V50', 'V51', 'V52']
target = 'V1'

# %%
# SEGMENTO 3: LIMPIEZA PROFUNDA DE DATOS
# Convertir todos los valores a números. Los textos o espacios se convierten en NaN
for col in variables_salud + [target]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar filas con valores faltantes (NaN)
df_model = df[variables_salud + [target]].dropna()

# Filtrar valores de no respuesta (8 y 9 suelen significar "No sabe/No contesta")
for col in variables_salud + [target]:
    df_model = df_model[df_model[col] <= 7]

# %%
# SEGMENTO 4: ONE-HOT ENCODING PARA V52 (VARIABLE BINARIA)
# Convertir V52 en columnas dummy (1 y 2 se convierten en columnas separadas)
df_model = pd.get_dummies(df_model, columns=['V52'], prefix='V52')

# Actualizar la lista de variables para incluir las nuevas columnas dummy
cols_v52 = [c for c in df_model.columns if c.startswith('V52_')]
variables_salud = [c for v in variables_salud for c in (cols_v52 if v == 'V52' else [v])]

# %%
# SEGMENTO 5: CREAR VARIABLE OBJETIVO (CLASES DE FELICIDAD)
def categorizar(val):
    """Convierte valores numéricos en categorías de felicidad"""
    if val <= 2: 
        return 'Feliz'
    elif val <= 4: 
        return 'Medianamente Feliz'
    else: 
        return 'Infeliz'

# Aplicar la función a todos los valores de la variable objetivo
df_model['Clase_Felicidad'] = df_model[target].apply(categorizar)

# %%
# SEGMENTO 6: DIVIDIR DATOS EN ENTRENAMIENTO Y PRUEBA
X = df_model[variables_salud]  # Variables independientes (características)
y = df_model['Clase_Felicidad']  # Variable objetivo (clases)

# Dividir: 80% entrenamiento, 20% prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# SEGMENTO 7: DEFINIR MODELOS SIMPLES
modelos = {
    'Decision Tree': DecisionTreeClassifier(max_depth=5, random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=3),
    'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
}


# %%
# SEGMENTO 8: ENTRENAR Y EVALUAR MODELOS
for nombre, modelo in modelos.items():
    # Entrenar el modelo con los datos de entrenamiento
    modelo.fit(X_train, y_train)
    
    # Hacer predicciones con los datos de prueba
    y_pred = modelo.predict(X_test)
    
    # Mostrar resultados
    print(f"\n--- RESULTADO CON {nombre.upper()} ---")
    print(f"Precisión: {accuracy_score(y_test, y_pred):.2%}")
    print(f"Margen de Error: {1 - accuracy_score(y_test, y_pred):.2%}")
    print("\nDesglose de aciertos por grupo:")
    print(classification_report(y_test, y_pred))

# %%
# SEGMENTO 9: SUGERENCIAS PARA MEJORAR PRECISIÓN
print("\n" + "="*60)
print("SUGERENCIAS PARA MEJORAR LA PRECISIÓN:")
print("="*60)
print("1. Balanceo de clases: class_weight='balanced' (ya aplicado)")
print("2. Ajustar hiperparámetros:")
print("   - Decision Tree: aumentar/disminuir max_depth")
print("   - KNN: cambiar n_neighbors (3, 5, 7)")
print("   - Logistic Regression: ajustar C (regularización)")
print("3. Selección de features: usar solo variables importantes")
print("4. Ensemble: combinar predicciones de varios modelos")
print("5. Cross-validation: validación cruzada para evaluación robusta")