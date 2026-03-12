import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier # Necesitarás instalarlo: pip install xgboost
from sklearn.metrics import classification_report, accuracy_score

# 1. Carga y Selección de Variables Extendida
df = pd.read_csv('3462_num.csv', sep=';', low_memory=False)

# Añadimos variables de contexto social y económico del PDF
vars_salud = ['V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V52']
vars_contexto = ['C_SATISFVIDA', 'DIFICULT_ECO', 'SEXO', 'DECADA', 'V10']
target = 'V1'

# Limpieza: Convertir a numérico y quitar 8s/9s (No sabe/No contesta)
cols_to_use = vars_salud + vars_contexto + [target]
for col in cols_to_use:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[~df[col].isin([8, 9])] # Limpiamos valores nulos de encuesta

df_model = df[cols_to_use].dropna()

# 2. Preparación (Ajustamos el target para que empiece en 0 para XGBoost)
X = df_model.drop(columns=[target])
y = df_model[target] - 1 # Si V1 va de 1 a 4, ahora irá de 0 a 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Modelo Potente: XGBoost con balanceo
model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=6,
    objective='multi:softprob',
    random_state=42
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"Nueva Precisión: {accuracy_score(y_test, y_pred):.2%}")
print(classification_report(y_test, y_pred))