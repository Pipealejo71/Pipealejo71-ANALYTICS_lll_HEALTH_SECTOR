import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE

df_final=("merged_df_2.csv")  
df_final=pd.read_csv(df_final)
df = df_final.copy()
df.dtypes

#------------------------------------#

# Verificar nombres de columnas
df_final.columns

# Estadísticas de los datos
df_final.describe(include = 'all').T

# Valores nulos
df_final.fillna('?', inplace=True)
column_null_counts = []

for column in df_final.columns:
    count = df_final[df_final[column] == '?'].shape[0]
    column_null_counts.append((column, count))
column_null_counts.sort()

for column, count in column_null_counts:
    print(column, count)
print('Se tienen', len(df_final['NRODOC'].unique()), 'pacientes únicos en los datos.')

# Cuántos pacientes se tienen?
print('Se tienen', len(df_final['NRODOC'].unique()), 'pacientes únicos en merged_df_2.')

# Análisis de edad vs CLASE FUNCIONAL
ax = sns.countplot(x="QUINQUENIO", hue="CLASE FUNCIONAL", data=df_final)
plt.xlabel('Age', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.show()

# EDAD VS Días hospitalizado
ax = sns.countplot(x='DIAS HOSPITALIZADO',   data= df_final)
plt.xlabel('QUINQUENIO', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.show()


# Cariables numericas y no numericas
numeric_vars = df.select_dtypes(include=[np.number])
non_numeric_vars = df.select_dtypes(exclude=[np.number])

# matrix de correlacion
corr_matrix = numeric_vars.corr()

plt.figure(figsize=(50, 20))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Matrix de Correlacion')
plt.show()

# Obtener los índices por encima de la diagonal
upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)

# Obtener las correlaciones por encima de la diagonal
upper_tri = corr_matrix.values[upper_tri_indices]

# Obtener los nombres de las columnas correspondientes
columns = [(corr_matrix.columns[i], corr_matrix.columns[j], upper_tri[n]) for n, (i, j) in enumerate(zip(*upper_tri_indices)) if upper_tri[n] > 0.8]

# Imprimir las columnas y sus correlaciones
for col1, col2, corr in columns:
    print(f"{col1} y {col2} tienen una correlación del {corr*100:.2f}%")

# Imprimir los tipos de datos en partes de 50 en 50 columnas
for i in range(0, len(df_final.columns), 50):
    print(df_final.dtypes[i:i+50])

# Crear listas de columnas por categoría
int_features = []
float_features = []
object_features = []

for column in df_final.columns:
    if df_final[column].dtype == 'int64':
        int_features.append(column)
    elif df_final[column].dtype == 'float64':
        float_features.append(column)
    elif df_final[column].dtype == 'object':
        object_features.append(column)

print("Integer Features:", int_features)
print("Float Features:", float_features)
print("Object Features:", object_features)

#----------------------------------------# 

# Crear un diccionario para almacenar los valores únicos de cada columna
unique_values = {}

# Iterar sobre cada columna del DataFrame
for column in df_final.columns:
    # Guardar los valores únicos de la columna en el diccionario
    unique_values[column] = df_final[column].unique()

# Mostrar los valores únicos de las columnas que tienen menos de 10 valores únicos
for column, values in unique_values.items():
    if len(values) < 10:
        print(f"Valores únicos en la columna {column} (total {len(values)}): {values}")


# Lista de columnas para convertir a categóricas
columns_to_convert = [
    "SERVICIO HABILITADO COD",    "SERVICIO HABILITADO",
    "REGIMEN AFILIACION",    "FUENTE FINANCIACION2",
    "SERVICIO ADMITE",    "BLOQUE ANTERIOR",
    "VIA INGRESO",    "BLOQUE",
    "UNIDAD ESTRATEGICA", "TIPO EGRESO",    "DEMORA SALIDA CLINICA (DIAS)",
    "TRANSFUSION SANGRE",    "ANTIBIOTICO",    "TIPO DIAGNOSTICO PRINCIPAL",    "CAUSA BASICA CAPITULO",
    "SEXO",    "MUNICIPIO",    "PRIMERA CLASE FUNCIONAL",
    "ÚLTIMA CLASE FUNCIONAL",    "CICLO_VITAL",
    "QUINQUENIO",    "YEAR",    "TIPO",    "CLASIFICACION IMC",    "AUTO-CALIFICACION NIVEL DE EJERCICIO",
    "CONSTANTES",    "CALIFICACION (INDICE DE FRAGILIDAD)",    "CALIFICACION (APOYO MONOPODAL)",
    "TIEMPO EN SEGUNDOS (RECORRER 5 METROS)",    "VELOCIDAD (M/S)",
    "CALIFICACION VELOCIDAD",    "TEST FINDRISC",    "INDICE TOBILLO/BRAZO",
    "DIABETES MELLITUS",    "TIPO DIABETES MELLITUS",    "CONTROL DIABETES",
    "TIENE RIESGO DE TENER DIABETES MELLITUS",    "TIENE HTA",
    "CONTROL HTA",    "TIENE RIESGO DE TENER HTA",
    "TIENE EPOC",
    "EPOC (CLASIFICACION BODEX)",    "ENFERMEDAD CORONARIA (EN EL ULTIMO AÑO)",
    "INSUFICIENCIA CARDIACA",    "VALVULOPATIA",    "ARRITMIA O PACIENTE CON DISPOSITIVO",
    "SUFRE DE ALGUNA ENFERMEDAD CARDIOVASCULAR",    "TABAQUISMO",
    "CLASIFICACION DE FRAMINGHAN",    "ESTADIO DE LA ENFERMEDAD RENAL",    "CLASE FUNCIONAL",
    "CLASIFICACION CAMBIO DE TFG",    "BASCILOSCOPIA",    "ULCERA DE PIE DIABETICO",    "REMISION",
    "TIENE PROXIMO CONTROL",    "REQUIERE CITA DE MORBILIDAD",    "AMBITO SEGUN EL MEDICO"
]
#
# Convertir cada columna en categórica
for column in columns_to_convert:
    if column in df_final.columns:  # Verificar si la columna existe en el DataFrame
        df_final[column] = df_final[column].astype('category')

# Verificar algunos resultados
print(df_final.dtypes)  # Imprime los tipos de datos para confirmar que son categóricos

""""
# Convertir variables categóricas en dummies
df_final_dummies = pd.get_dummies(df_final, drop_first=True)

# Mostrar las primeras filas para verificar los cambios
print(df_final_dummies.head())

#'DIAS HOSPITALIZADO' es la columna objetivo en DataFrame original
y = df_final['DIAS HOSPITALIZADO']
X = df_final_dummies.drop(['DIAS HOSPITALIZADO'], axis=1)

# Crear el modelo de regresión logística
model = LogisticRegression()

# Crear el selector RFE con el modelo logístico, especificando cuántas características deseamos conservar
rfe = RFE(model, n_features_to_select=10)

# Ajustar el RFE al conjunto de datos
rfe.fit(X, y)

# Ver qué características fueron seleccionadas
selected_features = X.columns[rfe.support_]
print("Características seleccionadas por RFE:", selected_features)
"""
# Convertir variables categóricas en dummies
df_final_dummies = pd.get_dummies(df_final, drop_first=True)

# Asegurarse de que 'DIAS HOSPITALIZADO' está en un formato adecuado y es numérico
df_final_dummies['DIAS HOSPITALIZADO'] = pd.to_numeric(df_final_dummies['DIAS HOSPITALIZADO'], errors='coerce')

# Separar el DataFrame en features y target
X = df_final_dummies.drop('DIAS HOSPITALIZADO', axis=1)
y = df_final_dummies['DIAS HOSPITALIZADO']

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# Crear el modelo de regresión logística
model = LogisticRegression(max_iter=1000)  # Aumentar el número de iteraciones si es necesario

# Crear el selector RFE con el modelo de regresión logística, especificando el número de características deseado
selector = RFE(model, n_features_to_select=10, step=1)  # Ajusta 'n_features_to_select' según tu criterio

# Ajustar el RFE al conjunto de datos de entrenamiento
selector.fit(X_train, y_train)

# Ver qué características fueron seleccionadas
selected_features = X.columns[selector.support_]
print("Características seleccionadas por RFE:", selected_features)