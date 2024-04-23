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
cronicos = cronicos.replace({'Ã³': 'o', 'Ã': 'u', 'Ã©': 'e', 'Ã¡': 'a'}, regex=True)
cronicos.columns = (cronicos.columns.str.replace('Ã³', 'o')
                                    .str.replace('Ã', 'u')
                                    .str.replace('Ã©', 'e')
                                    .str.replace('Ã¡', 'a')
                                    .str.replace('Ãº', 'u')
                                    .str.replace('Ã\x8d', 'i')
                                    .str.replace('Ã\xad', 'i')
                                    .str.replace('Ã±', 'ñ'))

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
#Se eliminan los datos de 2017 y 2018 ya se que son datos muy viejos 
usuarios = usuarios[usuarios['YEAR'] != '2017']
usuarios = usuarios[usuarios['YEAR'] != '2018']

## Egresos
egresos['YEAR'] = egresos['YEAR'].astype(str)
egresos['NRODOC'] = egresos['NRODOC'].astype(str)
egresos['NRO ATENCION'] = egresos['NRO ATENCION'].astype(str)
egresos['NRO INGRESO'] = egresos['NRO INGRESO'].astype(str)
egresos['SERVICIO HABILITADO COD'] = egresos['SERVICIO HABILITADO COD'].astype(str)

egresos['PERTINENCIA DIAGNOSTICA'].value_counts()
egresos = egresos.drop('PERTINENCIA DIAGNOSTICA', axis=1)
#Se eliminan los datos de 2017 y 2018 ya se que son datos muy viejos 
egresos = egresos[egresos['YEAR'] != '2017']
egresos = egresos[egresos['YEAR'] != '2018']

## Cronicos

cronicos['YEAR'] = cronicos['YEAR'].astype(str)
cronicos['NRODOC'] = cronicos['NRODOC'].astype(str)
cronicos['Atencion'] = cronicos['Atencion'].astype(str)
cronicos['Ingreso'] = cronicos['Ingreso'].astype(str)

cronicos['Es insulinorequiriente'].value_counts()
cronicos = cronicos.drop('Es insulinorequiriente', axis=1)
cronicos['Espirometria'].value_counts()
cronicos = cronicos.drop('Espirometria', axis=1)
cronicos['Otras Morbilidades'].value_counts()
cronicos = cronicos.drop('Otras Morbilidades', axis=1)
#Se eliminan todas las columnas de Diagnosticos ya que contienen mas del 70% de datos nulos
cronicos = cronicos.loc[:, ~cronicos.columns.str.contains('Diagnostico')]
#Se eliminan los datos de 2017 y 2018 ya se que son datos muy viejos 
cronicos = cronicos[cronicos['YEAR'] != '2017']
cronicos = cronicos[cronicos['YEAR'] != '2018']

cronicos.columns = cronicos.columns.str.upper()

#Se eliminan nulos de la columna ambito
# Crear una lista con los valores "Domiciliario" y "Ambulatorio"
values = ["Domiciliario", "Ambulatorio"]

# Seleccionar solo las filas donde "AMBITO SEGUN EL MEDICO" es "Domiciliario" o "Ambulatorio"
mask = cronicos["AMBITO SEGUN EL MEDICO"].isin(values)

# Invertir la selección y eliminar estas filas
cronicos = cronicos[mask]

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


#### Se eliminan las columnas que no aportan informacion relevante
# Calcular el porcentaje de valores nulos en cada columna
null_percent = merged_df_2.isnull().sum() / len(merged_df_2)
# Crear una lista de las columnas que tienen más del 95% de sus valores como nulos
columns_to_drop = null_percent[null_percent > 0.95].index
# Eliminar estas columnas del DataFrame
merged_df_2 = merged_df_2.drop(columns_to_drop, axis=1)

merged_df_2 = merged_df_2.drop(['TIPO CONTROL','TIPO IDENTIFICACION','YEAR','NRO ATENCION','NRO INGRESO','NRODOC','OBSERVACIONES','ALTA MEDICA','POSIBLE ALTA','ANALISIS Y CONDUCTA A SEGUIR'], axis=1)

#Exportacion
num_rows = len(merged_df_2)
merged_df_2 = merged_df_2.iloc[:split_index]
merged_df_2.to_csv('merged_df_2.csv', index=False)
merged_df_2.info()

### Codigo para Modelo





