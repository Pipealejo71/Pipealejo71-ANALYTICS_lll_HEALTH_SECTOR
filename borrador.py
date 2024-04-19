###FUNCIONES y LIBRERIAS


##LIBRERIAS
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
#pip install unidecode
import unidecode
import re
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import math



##FUNCIONES

# 1. Función para limpiar los nombres de las columnas
def Snake_case_clean(column_name):
    clean_name = unidecode.unidecode(column_name)
    clean_name = clean_name.lower()
    clean_name = clean_name.replace(' ', '_')
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_name)
    return clean_name



# 2. Función para observar las variables por tipo de datos
def Agrupar_variables_por_tipo(dataframe):
    variables_por_tipo = {}
    for column_name, dtype in dataframe.dtypes.items():
        tipo_str = str(dtype)
        if tipo_str not in variables_por_tipo:
            variables_por_tipo[tipo_str] = [column_name]
        else:
            variables_por_tipo[tipo_str].append(column_name)
    for tipo, columnas in variables_por_tipo.items():
        print(f"Tipo: {tipo}")
        for columna in columnas:
            print(f"  - {columna}")
        print()



# 3. Función para filtrar columnas por porcentaje
def Filtrar_columnas_por_porcentaje(df, porcentaje):
    porcentaje_no_nulos = df.count() / len(df)
    columnas_a_considerar = porcentaje_no_nulos[porcentaje_no_nulos >= porcentaje].index
    df = df[columnas_a_considerar]
    return df



# 4. Función para graficar las variables de usuarios
def Grafico1_usuarios(df):
    num_columnas = 4
    num_filas = math.ceil(len(usuarios.columns) / num_columnas)

    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(16, 12))
    plt.subplots_adjust(hspace=1.5)
    if num_filas == 1:
        axs = [axs]

    # Iterar sobre las columnas y dibujar los gráficos en los subgráficos
    for i, columna in enumerate(usuarios.columns):
        fila_actual = i // num_columnas
        columna_actual = i % num_columnas
        ax = axs[fila_actual][columna_actual]
        
        usuarios[columna].hist(ax=ax)
        ax.set_title('Distribución de {}'.format(columna))
        ax.set_xlabel(columna)
        ax.set_ylabel('Frecuencia')
        
        # Agregar el número de datos sobre cada barra
        for valor, frecuencia in usuarios[columna].value_counts().items():
            ax.text(valor, frecuencia, str(frecuencia), ha='center', va='bottom')
        ax.tick_params(axis='x', rotation=90)
    # Ocultar subgráficos no utilizados
    for i in range(len(usuarios.columns), num_filas * num_columnas):
        fila_actual = i // num_columnas
        columna_actual = i % num_columnas
        axs[fila_actual][columna_actual].axis('off')

    plt.tight_layout()
    plt.show()


#######
#######
#######
#######
#######
#######
#######
#######
#######


## Ruta directorio qué tiene paquetes
sys.path
#sys.path.append('C:\\Users\\User\\Desktop\\Analitica lll\\Pipealejo71-ANALYTICS_lll_HEALTH_SECTOR') 
sys.path.append('C:\\Users\\JavierBurgos\\Desktop\\LOCAL\\Analítica 3\\UNIDADES\\Unidad 3 - Aplicaciones en Operaciones (Salud)\\5.1 Entrega\\Pipealejo71-ANALYTICS_lll_HEALTH_SECTOR')

#Carga de datos
egresos=pd.read_csv('RETO_df_egresos.csv')
usuarios=pd.read_csv('RETO_df_usuarios.csv')
cronicos=pd.read_csv('RETO_df_cronicos.csv', delimiter=';', encoding='latin1')


#######
#######
#######
#######
#######
#######
#######
#######
#######


#PREPROCESAMIENTO


#USUARIOS

usuarios.info()
usuarios.shape

# Exclución de las personas que superan los 60 años
usuarios = usuarios[usuarios['EDAD'] > 60]

# 1. Función Snake_case_clean para limpiar y renombrar los nombres de las columnas
usuarios.rename(columns=Snake_case_clean, inplace=True)
display(usuarios.columns.tolist())

# 2 Función para agrupar las variables por tipo
Agrupar_variables_por_tipo(usuarios)

# 4. Función graficar distribución de los datos de cada variable en usuarios
Grafico1_usuarios(usuarios)

# Exclución de variables con unica categoria
usuarios = usuarios.drop(['departamento', 'ciclo_vital'], axis=1)

#
#


#
#
#

usuarios['YEAR'] = usuarios['YEAR'].astype(str)
usuarios['NRODOC'] = usuarios['NRODOC'].astype(str)



usuarios['PRIMERA CLASE FUNCIONAL'] = usuarios['PRIMERA CLASE FUNCIONAL'].str.upper()
usuarios['ÚLTIMA CLASE FUNCIONAL'] = usuarios['PRIMERA CLASE FUNCIONAL'].str.upper()
usuarios['PRIMERA CLASE FUNCIONAL'].value_counts()
usuarios['ÚLTIMA CLASE FUNCIONAL'].value_counts()

usuarios['SEXO'].value_counts()
usuarios = usuarios[usuarios['SEXO'].isin(['F', 'M'])]
usuarios['SEXO'].value_counts()

usuarios = usuarios[usuarios['EDAD'].notna()]

