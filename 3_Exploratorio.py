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

df_final=("merged_df_2.csv")  
df_final=pd.read_csv(df_final)
df = df_final.copy()
df.dtypes

# Cariables numericas y nonumericas
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