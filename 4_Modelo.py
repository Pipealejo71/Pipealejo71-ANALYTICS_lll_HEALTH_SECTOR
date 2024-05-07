import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor


# MODELOS
#Cargar paquetes

#Cargar data_base
df_final_2=("df_final_V2.csv")  
df_final_2=pd.read_csv(df_final_2)
df_2 = df_final_2.copy()
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

# Evaluación parametro MAE
mae_df_2 = medir_modelos(modelos, "neg_mean_absolute_error", X2, y2, 4)
mae_df_2.columns = ['dt', 'rf', 'gb']
mae_df_2


##Comparacion de metricas por modelo
# Tabla con promedio de métricas
comparacion_metricas_2 = pd.DataFrame({
    'Model': ['decision_tree', 'random_forest', 'gradient_boosting'],
    'MSE_Average': [-mse_df_2['dt'].mean(), -mse_df_2['rf'].mean(), -mse_df_2['gb'].mean()],
    'MAE_Average': [-mae_df_2['dt'].mean(), -mae_df_2['rf'].mean(), -mae_df_2['gb'].mean()]
})
comparacion_metricas_2

#------------------------------------#
###RANDOM FOREST####
### Tuneo de hiperparametros ###
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
tun_rf = RandomizedSearchCV(m_rf, param_distributions=param_grid, n_iter=20, scoring="neg_mean_absolute_error")

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
RF_mse = mean_squared_error(y_test, y_pred)

# Calcular el MAE
RF_mae = mean_absolute_error(y_test, y_pred)

# Imprimir el MSE y el MAE
print(f"MSE para RandomForest: {RF_mse}")
print(f"MAE para RandomForest: {RF_mae}")

#------------------------------------#

###GRADIENT BOOSTING####
### Tuneo de hiperparametros ###
# Definir los parámetros para la búsqueda
'''
Ajunte de hiperparametros

ATENTIORMENTE:
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [None, 3, 5, 10],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 3],
    'subsample': [0.5, 0.7, 1.0]
}

SE PASO DE:
    MSE para GradientBoosting: 3.7956224953798827
    MAE para GradientBoosting: 1.462416727459052
A:
    MSE para GradientBoosting: 0.5832093268592665
    MAE para GradientBoosting: 0.16097630876541147

'''
#Parametros testeados, para mejor resultado
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.5, 0.7, 0.9],
    'max_depth': [None, 3, 5, 7],
    'min_samples_split': [2, 5, 8],
    'min_samples_leaf': [1, 2, 3],
    'subsample': [0.5, 0.7, 0.9]
}

# Crear el objeto GradientBoostingRegressor
m_gb = GradientBoostingRegressor()

# Crear el objeto RandomizedSearchCV
tun_gb = RandomizedSearchCV(m_gb, param_distributions=param_grid, n_iter=20, scoring="neg_mean_absolute_error")

# Ajustar RandomizedSearchCV a los datos
tun_gb.fit(X2, y2)

# Imprimir los mejores parámetros
pd.set_option('display.max_colwidth', 100)
resultados = tun_gb.cv_results_
print(tun_gb.best_params_)
pd_resultados = pd.DataFrame(resultados)
display(pd_resultados[["params", "mean_test_score"]])

# Guardar el modelo con hyperparameter tunning
gb_final = tun_gb.best_estimator_

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.2, random_state=42)

# Ajustar el modelo al conjunto de entrenamiento
gb_final.fit(X_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = gb_final.predict(X_test)

# Calcular el MSE
GB_mse = mean_squared_error(y_test, y_pred)

# Calcular el MAE
GB_mae = mean_absolute_error(y_test, y_pred)

# Imprimir el MSE y el MAE
print(f"MSE para GradientBoosting: {GB_mse}")
print(f"MAE para GradientBoosting: {GB_mae}")