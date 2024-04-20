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

# 1. Función.1 para limpiar los nombres de las columnas
def Snake_case_clean(column_name):
    clean_name = unidecode.unidecode(column_name)
    clean_name = clean_name.lower()
    clean_name = clean_name.replace(' ', '_')
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_name)
    return clean_name



# 2. Función.2 para observar las variables por tipo de datos
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




# 3. Función.3 graficar distribución de los datos de cada variable en usuarios
def Grafico1_usuarios(df):
    num_columnas = 4
    num_filas = math.ceil(len(usuarios.columns) / num_columnas)

    fig, axs = plt.subplots(num_filas, num_columnas, figsize=(16, 12))
    plt.subplots_adjust(hspace=1.5)
    if num_filas == 1:
        axs = [axs]

    for i, columna in enumerate(usuarios.columns):
        fila_actual = i // num_columnas
        columna_actual = i % num_columnas
        ax = axs[fila_actual][columna_actual]
        
        usuarios[columna].hist(ax=ax)
        ax.set_title('Distribución de {}'.format(columna))
        ax.set_xlabel(columna)
        ax.set_ylabel('Frecuencia')
        
        for valor, frecuencia in usuarios[columna].value_counts().items():
            ax.text(valor, frecuencia, str(frecuencia), ha='center', va='bottom')
        ax.tick_params(axis='x', rotation=90)

    for i in range(len(usuarios.columns), num_filas * num_columnas):
        fila_actual = i // num_columnas
        columna_actual = i % num_columnas
        axs[fila_actual][columna_actual].axis('off')

    plt.tight_layout()
    plt.show()




# 4. Función.4 para filtrar columnas por porcentaje
def Filtrar_columnas_por_porcentaje(df, porcentaje):
    porcentaje_no_nulos = df.count() / len(df)
    columnas_a_considerar = porcentaje_no_nulos[porcentaje_no_nulos >= porcentaje].index
    df = df[columnas_a_considerar]
    return df


# 5. Fución.5 graficar distribución
def grafico_distribucion_variable(variable, df):

  if variable not in df.columns:
    raise ValueError(f"La variable '{variable}' no existe en el DataFrame.")

  datos = df[variable]

  tipo_dato = datos.dtype

  if tipo_dato in ['int64', 'float32', 'float64']:  # Si es numérico
    plt.figure(figsize=(10, 6))
    plt.hist(datos, bins='auto', edgecolor='black')
  elif tipo_dato == 'object':  # Si es categórica (texto)
    plt.figure(figsize=(10, 6))
    datos.value_counts().plot(kind='bar')
  else:  # Si es otro tipo (no compatible)
    raise TypeError(f"No se puede graficar la distribución de una variable de tipo '{tipo_dato}'.")

  if tipo_dato in ['int64', 'float32', 'float64']:
    plt.xlabel(variable)
    plt.ylabel('Frecuencia')
    plt.title(f"Distribución de {variable} en {df.shape[0]} registros")
    plt.grid(True)
  elif tipo_dato == 'object':
    plt.xlabel('Categoría')
    plt.ylabel('Frecuencia')
    plt.title(f"Distribución de {variable} en {df.shape[0]} registros")
    plt.xticks(rotation=45, ha='right')  # Rotar etiquetas de categorías para mejor visualización
    plt.grid(axis='y')  # Solo mostrar cuadrícula en el eje Y para barras

  plt.show()


# 6. Función.6 grafico conteo de datos por variable
def grafico_conteo_df(df):
    cantidad_total_datos = df.count()
    plt.figure(figsize=(10, 6))
    cantidad_total_datos.plot(kind='bar')
    plt.title('Cantidad total de datos por variable')
    plt.xlabel('Variables')
    plt.ylabel('Cantidad total de datos')
    plt.xticks(rotation=90)  
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


### 1. LECTURA DE DATOS
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


### 2. PREPROCESAMIENTO


## 2.1 USUARIOS

usuarios.info()
usuarios.shape

# 2.1.1 Filtro usuarios mayores a 60 años
usuarios = usuarios[usuarios['EDAD'] > 60]

# 2.1.2 Limpiar y renombrar las variables - Función.1
usuarios.rename(columns=Snake_case_clean, inplace=True)
############display(usuarios.columns.tolist())

# 2.1.3 Grafico. Distribución de cada variable - Función.3 
############Grafico1_usuarios(usuarios)

# 2.1.4 Filtro exclución de variables con unica categoria
usuarios = usuarios.drop(['departamento', 
                          'ciclo_vital'], axis=1)

# 2.1.5 Filtro datos asosiados a 2021
usuarios = usuarios[usuarios['year'] == 2021]

# 2.1.5.1 Grafico. Distribución de la variable mes
############grafico_distribucion_variable('barrio', usuarios)

# 2.1.6 Filtro datos asosiados a enero
usuarios = usuarios[usuarios['mes'] == 'ENERO']

# 2.1.7 Consulta de forma agrupada, las variables según su tipo - Función.2
Agrupar_variables_por_tipo(usuarios)

# 2.1.8 Reasignación de tipo de variable
usuarios['nrodoc'] = usuarios['nrodoc'].astype(str)

# 2.1.9 Filtro eliminar duplicados
usuarios = usuarios.drop_duplicates(subset=['nrodoc'], keep='first')

# 2.1.10 Filtro datos asosiados a MEDELLIN
usuarios = usuarios[usuarios['municipio'] == 'MEDELLIN']
# 2.1.11 Filtros 
usuarios = usuarios.drop(['municipio'
                          'year',
                          'mes'
                          'barrio'], axis=1)

# 2.1.12 PENDIENTE TRATAR LAS VARIABLES  CLASE FUNCIONAL
#   EJECUTAR EL SIGUIENTE CODIGO 
usuarios.info()

grafico_conteo_df(usuarios)










def grafico_distribucion_variable_barrio(df, variable):
    if variable not in df.columns:
        raise ValueError(f"La variable '{variable}' no existe en el DataFrame.")

    datos = df[variable].dropna()  # Handle missing values
    tipo_dato = datos.dtype

    if tipo_dato == 'object':  # Si es categórica (texto)
        datos_unicos = datos.unique().astype(str)
        # Check lengths for debugging
        print(f"Número de valores únicos: {len(datos_unicos)}")
        datos_conteo = pd.DataFrame({'valor': datos_unicos, 'frecuencia': datos.value_counts()})
        print(f"Largo de datos_conteo: {len(datos_conteo)}")
        plt.figure(figsize=(10, 6))
        datos_conteo.plot(x='valor', y='frecuencia', kind='bar', color='skyblue', edgecolor='black')
    else:  # Si es otro tipo (no compatible)
        raise TypeError(f"No se puede graficar la distribución de una variable de tipo '{tipo_dato}'.")

    plt.xlabel('Barrio')
    plt.ylabel('Frecuencia')
    plt.title(f"Distribución de barrios en {df.shape[0]} registros")
    plt.xticks(rotation=45, ha='right')  # Rotar etiquetas de categorías para mejor visualización
    plt.grid(axis='y')  # Solo mostrar cuadrícula en el eje Y para barras

    plt.show()

# Ejemplo de uso
grafico_distribucion_variable_barrio(usuarios, 'barrio')

