# MODELOS
#Cargar paquetes
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import cross_val_predict, cross_val_score, cross_validate, train_test_split 
from sklearn.metrics import mean_squared_error
from sklearn import metrics
import seaborn as sns
import scipy.stats as stats
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

#Cargar data_base
df_final=("df_final_V1.csv")
df_final_2=("df_final_V2.csv")  
df_final=pd.read_csv(df_final)
df_final_2=pd.read_csv(df_final_2)
df = df_final.copy()
df_2 = df_final_2.copy()
df.info()
df_2.info()

#------------------------------------#
# Función para medir modelos
def medir_modelos(modelos, scoring, X, y, cv):
    metric_modelos = pd.DataFrame()
    for modelo in modelos:
        scores = cross_val_score(modelo, X, y, scoring=scoring, cv=cv)
        pdscores = pd.DataFrame(scores)
        metric_modelos = pd.concat([metric_modelos, pdscores], axis=1)
    
    metric_modelos.columns = ["decision_tree", "random_forest", "gradient_boosting"]
    return metric_modelos

#------------------------------------#

#Modelos con variables seleccionadas DecisionTreeRegressor

X=df.drop(['DIAS HOSPITALIZADO'], axis=1)
y=df['DIAS HOSPITALIZADO']

##Definicion de Modelos 


m_rtree = DecisionTreeRegressor()  
m_rf = RandomForestRegressor()  
m_gbt = GradientBoostingRegressor()  

modelos=list([m_rtree, m_rf, m_gbt])

#------------------------------------#

# Modelo K fold cross validation
# Evaluación parametro MSE
mse_df = medir_modelos(modelos, "neg_mean_squared_error", X, y, 4)
mse_df.columns = ['dt', 'rf', 'gb']
mse_df

# Evaluación parametro R^2
r2_df = medir_modelos(modelos, "r2", X, y, 4)
r2_df.columns = ['dt', 'rf', 'gb']
r2_df

##Comparacion de metricas por modelo
# Tabla con promedio de métricas
comparacion_metricas = pd.DataFrame({
    'Model': ['decision_tree', 'random_forest', 'gradient_boosting'],
    'MSE_Average': [mse_df['dt'].mean(), mse_df['rf'].mean(), mse_df['gb'].mean()],
    'R2_Average': [ r2_df['dt'].mean(), r2_df['rf'].mean(), r2_df['gb'].mean()]
})
comparacion_metricas

#------------------------------------#
#Modelos con variables seleccionadas RandomForestRegressor

X2=df_2.drop(['DIAS HOSPITALIZADO'], axis=1)
y2=df_2['DIAS HOSPITALIZADO']

##Definicion de Modelos 

m_rtree = DecisionTreeRegressor()  
m_rf = RandomForestRegressor()  
m_gbt = GradientBoostingRegressor()  

modelos=list([m_rtree, m_rf, m_gbt])

#------------------------------------#

# Modelo K fold cross validation
# Evaluación parametro MSE
mse_df_2 = medir_modelos(modelos, "neg_mean_squared_error", X2, y2, 4)
mse_df_2.columns = ['dt', 'rf', 'gb']
mse_df_2

# Evaluación parametro R^2
r2_df_2 = medir_modelos(modelos, "r2", X2, y2, 4)
r2_df_2.columns = ['dt', 'rf', 'gb']
r2_df_2

##Comparacion de metricas por modelo
# Tabla con promedio de métricas
comparacion_metricas_2 = pd.DataFrame({
    'Model': ['decision_tree', 'random_forest', 'gradient_boosting'],
    'MSE_Average': [mse_df_2['dt'].mean(), mse_df_2['rf'].mean(), mse_df_2['gb'].mean()],
    'R2_Average': [ r2_df_2['dt'].mean(), r2_df_2['rf'].mean(), r2_df_2['gb'].mean()]
})
comparacion_metricas_2

#------------------------------------#
###ARBOLES DE DECISION####
### Tuneo de hiperparametros

#Para Arboles de decisión
param_grid = [{'max_depth': [2,15, None], 
               'min_samples_split': [10, 101, 10], 
               'min_samples_leaf': [1, 5, 10]}]

tun_rtree = RandomizedSearchCV(m_rtree, param_distributions=param_grid, n_iter=20, scoring="r2")
tun_rtree.fit(X2, y2)

pd.set_option('display.max_colwidth', 100)
resultados = tun_rtree.cv_results_
print(tun_rtree.best_params_)
pd_resultados = pd.DataFrame(resultados)
display(pd_resultados[["params", "mean_test_score"]])
rtree_final = tun_rtree.best_estimator_ ### Guardar el modelo con hyperparameter tunning



# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.2, random_state=42)

# Ajustar el modelo al conjunto de entrenamiento
rtree_final.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = rtree_final.predict(X_test)

# Calcular el MSE
mse = mean_squared_error(y_test, y_pred)

# Calcular el R^2
r2 = r2_score(y_test, y_pred)

# Imprimir el MSE y el R^2
print(f"MSE: {mse}")
print(f"R^2: {r2}")

#------------------------------------#
###RANDOM FOREST####
### Tuneo de hiperparametros
'''
Cambio de hiperparametros con espaciado estandar.  
Mejoró el desempeño

HIPERPARAMETROS ANTERIORES:

param_grid = {

    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'bootstrap': [True, False]
}

Se pasó de 
    MSE: 2.0322960454594607
    R^2: 0.9214058016555424

a
    MSE: 1.9898731429574636
    R^2: 0.9230464061437708
'''

# Definir los parámetros para la búsqueda
param_grid = {

    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 3],
    'bootstrap': [True, False]
}

# Crear el objeto RandomForestRegressor
m_rf = RandomForestRegressor()

# Crear el objeto RandomizedSearchCV
tun_rf = RandomizedSearchCV(m_rf, param_distributions=param_grid, n_iter=20, scoring="r2")

# Ajustar RandomizedSearchCV a los datos
tun_rf.fit(X2, y2)

# Imprimir los mejores parámetros
pd.set_option('display.max_colwidth', 100)
resultados = tun_rf.cv_results_
print(tun_rf.best_params_)
pd_resultados = pd.DataFrame(resultados)
display(pd_resultados[["params", "mean_test_score"]])

# Guardar el modelo con hyperparameter tunning
rf_final = tun_rf.best_estimator_

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Ajustar el modelo al conjunto de entrenamiento
rf_final.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = rf_final.predict(X_test)

# Calcular el MSE
mse = mean_squared_error(y_test, y_pred)

# Calcular el R^2
r2 = r2_score(y_test, y_pred)

# Imprimir el MSE y el R^2
print(f"MSE: {mse}")
print(f"R^2: {r2}")