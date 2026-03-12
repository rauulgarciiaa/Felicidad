import pandas as pd

df = pd.read_csv('3462_num.csv', sep=';', low_memory=False)

variables_salud = ['V24', 'V32', 'V33', 'V34', 'V35', 'V36', 'V37', 'V49', 'V52']
target = 'V1'

columnas = variables_salud + [target]

for col in columnas:
    df[col] = pd.to_numeric(df[col], errors='coerce')

df_salud = df[columnas].dropna()

for col in columnas:
    df_salud = df_salud[df_salud[col] <= 7]

df_salud.to_csv('datos_salud.csv', index=False, sep=';')
print(f"CSV generado: datos_salud.csv ({len(df_salud)} filas, {len(df_salud.columns)} columnas)")
print(f"Columnas: {list(df_salud.columns)}")
