#pip install unidecode
import unidecode
import re

# Función para limpiar los nombres de las columnas
def clean_column_name(column_name):
    # Reemplazar caracteres especiales y corregir errores de codificación
    clean_name = unidecode.unidecode(column_name)
    # Convertir a minúsculas
    clean_name = clean_name.lower()
    # Reemplazar espacios por guiones bajos
    clean_name = clean_name.replace(' ', '_')
    # Eliminar caracteres especiales restantes que no sean alfanuméricos o guiones bajos
    clean_name = re.sub(r'[^a-zA-Z0-9_]', '', clean_name)
    return clean_name

import pandas as pd

def agrupar_variables_por_tipo(dataframe):
    # Crear un diccionario para almacenar las variables agrupadas por tipo
    variables_por_tipo = {}

    # Iterar sobre las columnas del DataFrame
    for column_name, dtype in dataframe.dtypes.items():
        # Obtener el tipo de datos como una cadena
        tipo_str = str(dtype)
        # Agregar la columna al diccionario, agrupada por su tipo de datos
        if tipo_str not in variables_por_tipo:
            variables_por_tipo[tipo_str] = [column_name]
        else:
            variables_por_tipo[tipo_str].append(column_name)

    # Imprimir el diccionario
    for tipo, columnas in variables_por_tipo.items():
        print(f"Tipo: {tipo}")
        for columna in columnas:
            print(f"  - {columna}")
        print()
