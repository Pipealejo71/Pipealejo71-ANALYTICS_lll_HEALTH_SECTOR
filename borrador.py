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




# 2. Función.2 grafica distribución de los datos de cada variable en usuarios
def Grafico_distribucion_usuarios(df):
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


# 3. Fución.3 grafica distribución de una variable de un DataFrame
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



# 4. Función.4 Consulta agrupa las variables por tipo
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




# 4. Función.4 para filtrar columnas por porcentaje
def Filtrar_columnas_por_porcentaje(df, porcentaje):
    porcentaje_no_nulos = df.count() / len(df)
    columnas_a_considerar = porcentaje_no_nulos[porcentaje_no_nulos >= porcentaje].index
    df = df[columnas_a_considerar]
    return df


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
## Ruta directorio a bases
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


# PREPROCESAMIENTO

# 1 USUARIOS
#usuarios.info()
#usuarios.shape

# 1.1 Filtro usuarios mayores a 60 años
usuarios = usuarios[usuarios['EDAD'] > 60]

# 1.2 Limpiar y renombrar las variables - Función.1
usuarios.rename(columns=Snake_case_clean, inplace=True)
#display(usuarios.columns.tolist())

# 1.3 Consulta grafico distribución de cada variable - Función.2 
#Grafico_distribucion_usuarios(usuarios)

# 1.4 Filtro excluir de variables con única categoría
usuarios = usuarios.drop(['departamento', 
                          'ciclo_vital'], axis=1)

# 1.5 Filtro datos asociados a 2021
usuarios = usuarios[usuarios['year'] == 2021]

# 1.5.1 Consulta grafico distribución de variable 'mes' - Función.3
#grafico_distribucion_variable('mes', usuarios)

# 6 Filtro datos asociados a enero
usuarios = usuarios[usuarios['mes'] == 'ENERO']

# 1.7 Consulta agrupar las variables por tipo - Función.4
#Agrupar_variables_por_tipo(usuarios)

# 1.8 Reasignación de tipo a variables
usuarios['nrodoc'] = usuarios['nrodoc'].astype(str)
usuarios['edad'] = usuarios['edad'].astype('Int64')
usuarios['fecha_inicio_al_pgp'] = pd.to_datetime(usuarios['fecha_inicio_al_pgp'])
usuarios['fecha_primera_clase_funcional'] = pd.to_datetime(usuarios['fecha_primera_clase_funcional'])
usuarios['fecha_ultima_clase_funcional'] = pd.to_datetime(usuarios['fecha_ultima_clase_funcional'])

# 1.9 Filtro excluir duplicados
usuarios = usuarios.drop_duplicates(subset=['nrodoc'], keep='first')

# 1.10 Filtro datos asociados a MEDELLIN
usuarios = usuarios[usuarios['municipio'] == 'MEDELLIN']

# 1.11 Filtro excluir variables
usuarios = usuarios.drop(['municipio',
                          'year',
                          'mes',
                          'barrio'], axis=1)

# 1.12 Verifica categorías de cada variable
## Consultas
#usuarios.info()
#usuarios['sexo'].value_counts(dropna=False)
#usuarios['primera_clase_funcional'].value_counts(dropna=False)
#usuarios['ultima_clase_funcional'].value_counts(dropna=False)
#usuarios['quinquenio'].value_counts(dropna=False)
## Depuración
usuarios = usuarios[usuarios['fecha_primera_clase_funcional'].notna()]
usuarios['primera_clase_funcional'] = usuarios['primera_clase_funcional'].str.upper()
usuarios['ultima_clase_funcional'] = usuarios['ultima_clase_funcional'].str.upper()

# 1.13 Consulta final en usuarios
usuarios.reset_index(inplace=True, drop=True)
usuarios.info()



# 2 EGRESOS
#egresos.info()
#egresos.shape




# 2 CRONICOS




# EXPORTAR BASE a CSV
#BASE.to_csv('BASE.csv', index=False)