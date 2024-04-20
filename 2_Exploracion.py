import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys


## Ruta directorio qué tiene paquetes
sys.path
sys.path.append('C:\\Users\\User\\Desktop\\Analitica lll\\Pipealejo71-ANALYTICS_lll_HEALTH_SECTOR') 
#sys.path.append('C:\\Users\\JavierBurgos\\Desktop\\LOCAL\\Analítica 3\\UNIDADES\\Unidad 3 - Aplicaciones en Operaciones (Salud)\\5.1 Entrega\\Pipealejo71-ANALYTICS_lll_HEALTH_SECTOR')

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

usuarios = usuarios[usuarios['EDAD'].notna()]
usuarios = usuarios[usuarios['EDAD'] > 60]

## Egresos
egresos['YEAR'] = egresos['YEAR'].astype(str)
egresos['NRODOC'] = egresos['NRODOC'].astype(str)
egresos['NRO ATENCION'] = egresos['NRO ATENCION'].astype(str)
egresos['NRO INGRESO'] = egresos['NRO INGRESO'].astype(str)
egresos['SERVICIO HABILITADO COD'] = egresos['SERVICIO HABILITADO COD'].astype(str)

egresos['PERTINENCIA DIAGNOSTICA'].value_counts()
egresos = egresos.drop('PERTINENCIA DIAGNOSTICA', axis=1)

## Cronicos

cronicos['YEAR'] = cronicos['YEAR'].astype(str)
cronicos['NRODOC'] = cronicos['NRODOC'].astype(str)
cronicos = cronicos.rename(columns={'AtenciÃ³n': 'Atencion'})
cronicos['Atencion'] = cronicos['Atencion'].astype(str)
cronicos['Ingreso'] = cronicos['Ingreso'].astype(str)

cronicos['Es insulinorequiriente'].value_counts()
cronicos = cronicos.drop('Es insulinorequiriente', axis=1)
cronicos['Espirometria'].value_counts()
cronicos = cronicos.drop('Espirometria', axis=1)
cronicos['Otras Morbilidades'].value_counts()
cronicos = cronicos.drop('Otras Morbilidades', axis=1)

##Union de BD

#Union de Bases Egresos y Usuarios por NRODOC 
merged_df = pd.merge(egresos, usuarios, on='NRODOC', how='inner')
merged_df['YEAR_y'] = merged_df['YEAR_x']
merged_df = merged_df.drop('YEAR_x', axis=1)
merged_df = merged_df.rename(columns={'YEAR_y': 'YEAR'})

merged_df['MES_y'] = merged_df['MES_x']
merged_df = merged_df.drop('MES_x', axis=1)
merged_df = merged_df.rename(columns={'MES_y': 'MES'})

merged_df['FECHA NACIMIENTO_y'] = merged_df['FECHA NACIMIENTO_x']
merged_df = merged_df.drop('FECHA NACIMIENTO_x', axis=1)
merged_df = merged_df.rename(columns={'FECHA NACIMIENTO_y': 'FECHA NACIMIENTO'})
merged_df.columns

#Inner Join de Base merged_df con Cronicos por NRODOC 
merged_df_2 = pd.merge(merged_df, cronicos, on='NRODOC', how='inner')
merged_df_2['YEAR_y'] = merged_df_2['YEAR_x']
merged_df_2 = merged_df_2.drop('YEAR_x', axis=1)
merged_df_2 = merged_df_2.rename(columns={'YEAR_y': 'YEAR'})

merged_df_2['MES_y'] = merged_df_2['MES_x']
merged_df_2 = merged_df_2.drop('MES_x', axis=1)
merged_df_2 = merged_df_2.rename(columns={'MES_y': 'MES'})

#Se eliminan todas las columnas de Diagnosticos ya que contienen mas del 70% de datos nulos
merged_df_2 = merged_df_2.loc[:, ~merged_df_2.columns.str.contains('Diagnostico')]

#Esta base final contendra la union de las tres bases con los filtros de los usuarios que tengan mas de 60 años y esten en la base de cronicos




#### Cambios de Javier #######
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

# cronicos_filtrado ahora contiene solo las columnas con el 80% o más de sus datos diferentes de nulos
cronicos=filtrar_columnas_por_porcentaje(cronicos, 0.8)


