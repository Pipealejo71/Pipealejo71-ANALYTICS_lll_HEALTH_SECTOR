import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


## Ruta directorio qué tiene paquetes
sys.path
#sys.path.append('C:\\Users\\User\\Desktop\\Analitica lll\\Pipealejo71-ANALYTICS_lll_HEALTH_SECTOR') ## este comanda agrega una ruta
sys.path.append('C:\\Users\\JavierBurgos\\Desktop\\LOCAL\\Analítica 3\\UNIDADES\\Unidad 3 - Aplicaciones en Operaciones (Salud)\\5.1 Entrega\\Pipealejo71-ANALYTICS_lll_HEALTH_SECTOR')

#Carga de datos
egresos=pd.read_csv('RETO_df_egresos.csv')
usuarios=pd.read_csv('RETO_df_usuarios.csv')
cronicos=pd.read_csv('RETO_df_cronicos.csv', delimiter=';', encoding='latin1')

#Exploración de datos
egresos.info()
egresos.shape
usuarios.info()
usuarios.shape
cronicos.info()
cronicos.shape

#Preprocesamiento de datos
## Usuario
usuarios['YEAR'] = usuarios['YEAR'].astype(str)
usuarios['NRODOC'] = usuarios['NRODOC'].astype(str)

usuarios['DEPARTAMENTO'].value_counts()
usuarios = usuarios.drop('DEPARTAMENTO', axis=1)

usuarios['PRIMERA CLASE FUNCIONAL'] = usuarios['PRIMERA CLASE FUNCIONAL'].str.upper()
usuarios['ÚLTIMA CLASE FUNCIONAL'] = usuarios['PRIMERA CLASE FUNCIONAL'].str.upper()
usuarios['PRIMERA CLASE FUNCIONAL'].value_counts()
usuarios['ÚLTIMA CLASE FUNCIONAL'].value_counts()

usuarios['SEXO'].value_counts()
usuarios = usuarios[usuarios['SEXO'].isin(['F', 'M'])]
usuarios['SEXO'].value_counts()

## Egresos
egresos['YEAR'] = egresos['YEAR'].astype(str)
egresos['NRODOC'] = egresos['NRODOC'].astype(str)
egresos['NRO ATENCION'] = egresos['NRO ATENCION'].astype(str)
egresos['NRO INGRESO'] = egresos['NRO INGRESO'].astype(str)
egresos['SERVICIO HABILITADO COD'] = egresos['SERVICIO HABILITADO COD'].astype(str)

egresos['PERTINENCIA DIAGNOSTICA'].value_counts()
egresos = egresos.drop('PERTINENCIA DIAGNOSTICA', axis=1)

## Cronicos

# Función para renombrar las columnas
cronicos.rename(columns=clean_column_name, inplace=True)
display(cronicos.columns.tolist())

# Función para observar las variables por tipo de datos
agrupar_variables_por_tipo(cronicos)

#gráfico
import matplotlib.pyplot as plt
# Contar la cantidad total de datos en cada variable
cantidad_total_datos = cronicos.count()
# Crear el gráfico de barras
plt.figure(figsize=(10, 6))
cantidad_total_datos.plot(kind='bar')
plt.title('Cantidad total de datos por variable')
plt.xlabel('Variables')
plt.ylabel('Cantidad total de datos')
plt.xticks(rotation=90)  # Rotar las etiquetas del eje x para una mejor legibilidad
plt.tight_layout()
plt.show()

#Cantidad de datos en cada variable
cantidad_total_datos = cronicos.count()
print(cantidad_total_datos)
cronicos.describe
# Identificar variables vacías
variables_vacias = cronicos.isnull().any()
print(variables_vacias)
# Eliminar columnas vacías
cronicos = cronicos.dropna(axis=1)
#Eliminar duplicados

#eliminar variables con menos de 1000 datos



