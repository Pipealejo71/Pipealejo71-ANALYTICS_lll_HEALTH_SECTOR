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

