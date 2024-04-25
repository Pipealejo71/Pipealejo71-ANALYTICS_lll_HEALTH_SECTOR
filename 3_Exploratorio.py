#Cargar paquetes
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
import scipy.stats as stats
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import chi2_contingency

#Cargar data_base
df_final=("merged_df_2.csv")  
df_final=pd.read_csv(df_final)
df_copy = df_final.copy()


#------------------------------------#

# Verificar nombres de columnas
df_final.columns

# Estadísticas de los datos
df_copy.describe(include = 'all').T

# Valores nulos
df_copy.fillna('?', inplace=True)
column_null_counts = []

for column in df_copy.columns:
    count = df_copy[df_copy[column] == '?'].shape[0]
    column_null_counts.append((column, count))
column_null_counts.sort()

for column, count in column_null_counts:
    print(column, count)
print('Se tienen', len(df_copy['NRODOC'].unique()), 'pacientes únicos en los datos.')

# Cuántos pacientes se tienen?
print('Se tienen', len(df_copy['NRODOC'].unique()), 'pacientes únicos en merged_df_2.')

# Análisis de edad vs CLASE FUNCIONAL
ax = sns.countplot(x="QUINQUENIO", hue="CLASE FUNCIONAL", data=df_copy)
plt.xlabel('Age', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.show()

# EDAD VS Días hospitalizado
ax = sns.countplot(x='DIAS HOSPITALIZADO',   data= df_copy)
plt.xlabel('QUINQUENIO', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Count', size = 12)
plt.show()

#Cantidad de pacientes por clase funcional
# Contar los valores únicos de 'NRODOC' en cada 'CLASE FUNCIONAL'
df_grouped = df_copy.groupby('CLASE FUNCIONAL')['NRODOC'].nunique().reset_index()
# Crear la gráfica de barras
ax = sns.barplot(x='CLASE FUNCIONAL', y='NRODOC', data=df_grouped)
plt.xlabel('Clase Funcional', size = 12)
plt.xticks(rotation=90, size = 12)
plt.ylabel('Cantidad de Pacientes Únicos', size = 12)
plt.show()

###
df_final = df_final[df_final['CLASE FUNCIONAL'] == 'Clase funcional 2A']
'''
Se realizara una prediccion de los pacientes
con clase funcional 2A ya que es la clase 
con el mayor numero de pacientes registrada
'''

###

# Variables numericas y no numericas
numeric_vars = df_final.select_dtypes(include=[np.number])
non_numeric_vars = df_final.select_dtypes(exclude=[np.number])


### MATRIX DE CORRELACION VARIABLES NUMERICAS
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

column_names = [col1 for col1, col2, corr in columns]

### Evaluacion Chi-cuadrado para evaluar colinealidad de variables categoricas
# Crear una lista para almacenar las columnas categóricas
categorical_columns = []

# Iterar sobre cada columna del DataFrame
for column in df_final.columns:
    # Guardar los valores únicos de la columna
    unique_values = df_final[column].unique()

    # Si la columna tiene más de 1 valor único pero menos de 10, convertirla a categórica
    if 1 < len(unique_values) < 10:
        print(f"Valores únicos en la columna {column} (total {len(unique_values)}): {unique_values}")
        df_final[column] = df_final[column].astype('category')
        categorical_columns.append(column)

# Crear un nuevo DataFrame con las columnas categóricas
df_categoricas = df_final[categorical_columns]
df_categoricas.dtypes

# Crear una lista para almacenar los pares de columnas con colinealidad y sus valores p
collinear_pairs = []

# Iterar sobre cada par de columnas
for i, column1 in enumerate(df_categoricas.columns):
    for j in range(i + 1, len(df_categoricas.columns)):
        column2 = df_categoricas.columns[j]

        # Crear una tabla de contingencia
        contingency_table = pd.crosstab(df_categoricas[column1], df_categoricas[column2])

        # Realizar la prueba de chi-cuadrado
        chi2, p, dof, expected = chi2_contingency(contingency_table)

        # Si el p-valor es menor que 0.05, entonces las columnas están asociadas
        if p < 0.05:
            collinear_pairs.append((column1, column2, p))


### ELIMINACION DE COLUMNAS ###

# Eliminar las columnas altamente correlacionadas para evitar multicolinealidad
df_final = df_final.drop(column_names, axis=1)

#Eliminar columnas con corre

# Eliminar columnas con valores constantes
df_final = df_final.loc[:, df_final.apply(pd.Series.nunique) != 1]

# Contar columnas con valores nulos
null_columns = df_final.columns[df_final.isnull().any()]
len(null_columns)

# Imprimir el conteo de valores nulos por columna
for column in null_columns:
    print(f"{column}: {df_final[column].isnull().sum()}")

# Eliminar columnas con valores nulos
df_final = df_final.drop(null_columns, axis=1)

#Seleccion y filtrado de columnas 
#----------------------------------------# 
###SELECCION VARIABLES ###


y = df_final['DIAS HOSPITALIZADO']
df_corr=df_final.drop(['INGRESO','NRO INGRESO','NRODOC','DIAS HOSPITALIZADO'], axis=1)
df_corr.columns
# Convertir variables categóricas en dummies
df_corr_dummies = pd.get_dummies(df_categoricas)

#Normalizar variables numericas
df_final_sel = df_final.select_dtypes(include = ["number"]) # filtrar solo variables númericas
df_final_sel = df_final_sel.drop(['DIAS HOSPITALIZADO'], axis=1)
df_final_V2_norm = df_final_sel.copy(deep = True)  # crear una copia del DataFrame
scaler = MinMaxScaler()  # asignar el tipo de normalización
sv = scaler.fit_transform(df_final_V2_norm.iloc[:, :])  # normalizar los datos
df_final_V2_norm.iloc[:, :] = sv  # asignar los nuevos datos
df_final_V2_norm.head()

# Se unen la variables numericas y categoricas
df_final_corr = pd.concat([df_final_V2_norm, df_corr_dummies], axis=1)
#----------------------------------------#
### Seleccion de variables mediante Arbol de decision###

# Crear un árbol de decisión con parámetros ajustados
tree = DecisionTreeRegressor(max_depth=10, min_samples_split=10, min_samples_leaf=5, random_state=0)

# Ajustar el árbol a tus datos
tree.fit(df_final_corr, y)

# Obtener la importancia de las características
importances = tree.feature_importances_

# Crear una lista de tuplas (importancia, columna)
importances = [(round(importance, 5), column) for importance, column in zip(importances, df_final_corr.columns)]

# Ordenar la lista en función de la importancia
importances.sort(reverse=True)
len(importances)
#----------------------------------------#
###Seleccion de variables mediante Random Forest###

# Crear un Random Forest
forest = RandomForestRegressor(random_state=0)

# Ajustar el Random Forest a tus datos
forest.fit(df_final_corr, y)

# Obtener la importancia de las características
importances_2 = forest.feature_importances_

# Crear una lista de tuplas (importancia, columna)
importances_2 = [(round(importances_2, 5), column) for importances_2, column in zip(importances_2, df_final_corr.columns)]
# Ordenar la lista en función de la importancia
importances_2.sort(reverse=True)

#Lista de variables a usar en modelos
top_1= [column for importances, column in importances[:20]]
top_2= [column for importances_2, column in importances_2[:20]]

#creacion de dataframe y exportacion de BD
df_final_V1 = df_final_corr[top_1]
df_final_V2 = df_final_corr[top_2]
df_final_V1['DIAS HOSPITALIZADO'] = df_final['DIAS HOSPITALIZADO']
df_final_V2['DIAS HOSPITALIZADO'] = df_final['DIAS HOSPITALIZADO']
#Exportacion
df_final_V1.to_csv('df_final_V1.csv', index=False)
df_final_V2.to_csv('df_final_V2.csv', index=False)


