import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

### Ruta directorio qué tiene paquetes
sys.path
sys.path.append('C:\\Users\\User\\Desktop\\Analitica lll\\Pipealejo71-ANALYTICS_lll_HEALTH_SECTOR') ## este comanda agrega una ruta
#Carga de datos
egresos = ('RETO_df_egresos.csv')
usuario = ('RETO_df_usuarios.csv')
egresos=pd.read_csv(egresos)
usuario=pd.read_csv(usuario)

#Exploración de datos
egresos.info()
egresos.shape
usuario.info()
usuario.shape

#Preprocesamiento de datos
## Usuario
usuario['YEAR'] = usuario['YEAR'].astype(str)
usuario['NRODOC'] = usuario['NRODOC'].astype(str)

usuario['DEPARTAMENTO'].value_counts()
usuario = usuario.drop('DEPARTAMENTO', axis=1)

usuario['PRIMERA CLASE FUNCIONAL'] = usuario['PRIMERA CLASE FUNCIONAL'].str.upper()
usuario['ÚLTIMA CLASE FUNCIONAL'] = usuario['PRIMERA CLASE FUNCIONAL'].str.upper()
usuario['PRIMERA CLASE FUNCIONAL'].value_counts()
usuario['ÚLTIMA CLASE FUNCIONAL'].value_counts()

usuario['SEXO'].value_counts()
usuario = usuario[usuario['SEXO'].isin(['F', 'M'])]
usuario['SEXO'].value_counts()

## Egresos
egresos['YEAR'] = egresos['YEAR'].astype(str)
egresos['NRODOC'] = egresos['NRODOC'].astype(str)
egresos['NRO ATENCION'] = egresos['NRO ATENCION'].astype(str)
egresos['NRO INGRESO'] = egresos['NRO INGRESO'].astype(str)
egresos['SERVICIO HABILITADO COD'] = egresos['SERVICIO HABILITADO COD'].astype(str)

egresos['PERTINENCIA DIAGNOSTICA'].value_counts()
egresos = egresos.drop('PERTINENCIA DIAGNOSTICA', axis=1)