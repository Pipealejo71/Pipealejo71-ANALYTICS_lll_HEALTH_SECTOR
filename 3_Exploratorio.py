import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif

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


### Seleccion de variables por metodo Wrapper ###
#Backward selection
df_final_sel = df_final.select_dtypes(include = ["number"]) # filtrar solo variables númericas
#df_final_V2_int = df_final_V2_int.drop(['Attrition', 'retirementDate'], axis = 1) # excluir 'Attrition' y 'retirementDate'
y = df_final['DIAS HOSPITALIZADO']
df_final_sel.head()