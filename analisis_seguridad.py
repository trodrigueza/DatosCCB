import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set plot styles for better aesthetics
plt.style.use('ggplot')
sns.set_style('whitegrid')
sns.set_context('talk')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

# Colors for visualization
colors = px.colors.qualitative.Plotly

# Create output directory for visualizations
os.makedirs('visualizaciones', exist_ok=True)

print("Iniciando análisis de datos de seguridad y criminalidad...")

# Function to load and clean datasets
def load_dataset(filename):
    print(f"Cargando {filename}...")
    try:
        # Intentar cargar con punto y coma como delimitador
        df = pd.read_csv(f"datos/{filename}", encoding='latin1', low_memory=False, sep=';')
        
        # Verificar si los datos se cargaron en una sola columna
        if df.shape[1] == 1 and ',' in df.iloc[0, 0]:
            # Si es así, dividir la columna
            column_name = df.columns[0]
            new_df = df[column_name].str.split(',', expand=True)
            
            # Extraer los nombres de las columnas de la primera fila si contiene encabezados
            if ',' in column_name:
                headers = column_name.split(',')
                new_df.columns = headers
            
            df = new_df
        
        # Filtrar datos entre 2010 y 2024 si hay columna de año
        posibles_cols_anio = [col for col in df.columns if 'AÃ±o' in col or 'Año' in col or 'ANO' in col.upper() or 'YEAR' in col.upper()]
        
        if posibles_cols_anio:
            anio_col = posibles_cols_anio[0]
            print(f"  Filtrando datos entre 2010 y 2024 usando columna {anio_col}")
            
            # Convertir a numérico y filtrar
            df[anio_col] = pd.to_numeric(df[anio_col], errors='coerce')
            df = df[(df[anio_col] >= 2010) & (df[anio_col] <= 2024)]
        elif 'FECHA' in df.columns:
            print(f"  Filtrando datos entre 2010 y 2024 usando columna FECHA")
            # Convertir a datetime y filtrar
            df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
            df = df[(df['FECHA'].dt.year >= 2010) & (df['FECHA'].dt.year <= 2024)]
        
        print(f"  Filas después de filtrar por año: {df.shape[0]}, Columnas: {df.shape[1]}")
        return df
    except Exception as e:
        # Si falla, intentar con coma como delimitador
        try:
            df = pd.read_csv(f"datos/{filename}", encoding='latin1', low_memory=False)
            
            # Filtrar datos entre 2010 y 2024 si hay columna de año
            posibles_cols_anio = [col for col in df.columns if 'AÃ±o' in col or 'Año' in col or 'ANO' in col.upper() or 'YEAR' in col.upper()]
            
            if posibles_cols_anio:
                anio_col = posibles_cols_anio[0]
                print(f"  Filtrando datos entre 2010 y 2024 usando columna {anio_col}")
                
                # Convertir a numérico y filtrar
                df[anio_col] = pd.to_numeric(df[anio_col], errors='coerce')
                df = df[(df[anio_col] >= 2010) & (df[anio_col] <= 2024)]
            elif 'FECHA' in df.columns:
                print(f"  Filtrando datos entre 2010 y 2024 usando columna FECHA")
                # Convertir a datetime y filtrar
                df['FECHA'] = pd.to_datetime(df['FECHA'], errors='coerce')
                df = df[(df['FECHA'].dt.year >= 2010) & (df['FECHA'].dt.year <= 2024)]
            
            print(f"  Filas después de filtrar por año: {df.shape[0]}, Columnas: {df.shape[1]}")
            return df
        except Exception as e:
            print(f"Error al cargar {filename}: {e}")
            return None

# Helper function to get dataset summary
def dataset_summary(df, name):
    summary = {
        "nombre": name,
        "filas": df.shape[0],
        "columnas": df.shape[1],
        "columnas_datos": list(df.columns),
        "tipos_datos": {col: str(df[col].dtype) for col in df.columns},
        "valores_nulos": df.isnull().sum().sum(),
        "porcentaje_nulos": round((df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100, 2)
    }
    return summary

# ==========================================
# Cargar todos los datasets
# ==========================================

# Lista de archivos
archivos = [
    "Hurto_Personas.csv",
    "Capturas.csv",
    "Frentes_De_Seguridad.csv",
    "Hurto_Comercio.csv", 
    "Hurto_Automotores.csv",
    "Homicidios.csv",
    "Delitos_Informáticos.csv",
    "invasión_Usurpación_Tierras.csv",
    "Incautación_Estupefacientes.csv",
    "Delitos_Contra_Medio_Ambiente.csv",
    "Presupuesto_de_Gastos.csv",
    "Violencia_Intrafamiliar.csv"
]

# Diccionario para almacenar datasets
datasets = {}
summaries = []

# Cargar cada dataset
for archivo in archivos:
    nombre = archivo.replace(".csv", "")
    df = load_dataset(archivo)
    if df is not None:
        datasets[nombre] = df
        summaries.append(dataset_summary(df, nombre))

print("\nResumen de los datasets cargados:")
for summary in summaries:
    print(f"\n{summary['nombre']}:")
    print(f"  Filas: {summary['filas']}, Columnas: {summary['columnas']}")
    print(f"  Valores nulos: {summary['valores_nulos']} ({summary['porcentaje_nulos']}%)")
    print(f"  Columnas: {', '.join(summary['columnas_datos'][:5])}{'...' if len(summary['columnas_datos']) > 5 else ''}")

# ==========================================
# Análisis de tendencias temporales
# ==========================================

def analizar_tendencia_temporal(df, nombre, columna_fecha=None, columna_categoria=None):
    """Analiza y visualiza tendencias temporales en los datos"""
    try:
        print(f"\nAnalizando tendencias temporales en {nombre}...")
        
        # Caso 1: Columnas separadas de Año, Mes, Día
        if 'Año' in df.columns or 'AÑO' in df.columns:
            # Identificar la columna de año (puede variar en mayúsculas/minúsculas o tener acentos)
            columna_anio = next((col for col in df.columns if col.upper() in ['AÑO', 'ANO', 'YEAR']), None)
            
            if columna_anio:
                print(f"  Usando columna de año: {columna_anio}")
                # Asegurarse que es numérico
                df['año'] = pd.to_numeric(df[columna_anio], errors='coerce')
                
                # Crear agregación por año
                yearly_counts = df.groupby('año').size().reset_index(name='cantidad')
                
                # Visualización con Plotly
                fig = px.line(yearly_counts, x='año', y='cantidad', 
                             title=f'Tendencia Anual - {nombre}',
                             labels={'cantidad': 'Cantidad de Casos', 'año': 'Año'},
                             markers=True,
                             width=1200,  # Mayor ancho
                             height=700)  # Mayor altura
                
                fig.update_layout(
                    template='plotly_white',
                    legend_title_text='',
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=14),  # Fuente más grande
                    title=dict(font=dict(size=24)),  # Título más grande
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                
                fig.write_html(f'visualizaciones/{nombre}_tendencia_anual.html')
                
                # Si hay una columna de categoría, analizar tendencias por categoría
                if columna_categoria and columna_categoria in df.columns:
                    if df[columna_categoria].nunique() <= 10:  # Solo si hay un número razonable de categorías
                        # Crear copia del dataframe para no modificar el original
                        df_copy = df.copy()
                        
                        # Normalizar valores de género si corresponde
                        if columna_categoria.upper() in ['GENERO', 'GÉNERO', 'GÃ©NERO']:
                            # Normalizar valores no reportados
                            no_reportado = ['NO REPORTA', '-', 'NO REPORTADO', 'NO INFORMA', 'SIN INFORMACIÓN', 'DESCONOCIDO']
                            df_copy[columna_categoria] = df_copy[columna_categoria].replace(no_reportado, 'NO REPORTADO')
                        
                        category_yearly = df_copy.groupby(['año', columna_categoria]).size().reset_index(name='cantidad')
                        
                        fig = px.line(category_yearly, x='año', y='cantidad', color=columna_categoria,
                                     title=f'Tendencia Anual por {columna_categoria} - {nombre}',
                                     labels={'cantidad': 'Cantidad de Casos', 'año': 'Año'},
                                     markers=True,
                                     width=1200,  # Mayor ancho
                                     height=700)  # Mayor altura
                        
                        fig.update_layout(
                            template='plotly_white',
                            legend_title_text=columna_categoria,
                            plot_bgcolor='white',
                            font=dict(family="Arial", size=14),  # Fuente más grande
                            title=dict(font=dict(size=24)),  # Título más grande
                            xaxis=dict(showgrid=True, gridcolor='lightgray'),
                            yaxis=dict(showgrid=True, gridcolor='lightgray')
                        )
                        
                        fig.write_html(f'visualizaciones/{nombre}_tendencia_por_{columna_categoria}.html')
                
                return True
        
        # Caso 2: Una sola columna de fecha
        elif columna_fecha and columna_fecha in df.columns:
            print(f"  Usando columna de fecha: {columna_fecha}")
            # Convertir columna de fecha
            df[columna_fecha] = pd.to_datetime(df[columna_fecha], errors='coerce')
            df['año'] = df[columna_fecha].dt.year
            
            # Crear agregación por año
            yearly_counts = df.groupby('año').size().reset_index(name='cantidad')
            
            # Visualización con Plotly
            fig = px.line(yearly_counts, x='año', y='cantidad', 
                         title=f'Tendencia Anual - {nombre}',
                         labels={'cantidad': 'Cantidad de Casos', 'año': 'Año'},
                         markers=True,
                         width=1200,  # Mayor ancho
                         height=700)  # Mayor altura
            
            fig.update_layout(
                template='plotly_white',
                legend_title_text='',
                plot_bgcolor='white',
                font=dict(family="Arial", size=14),  # Fuente más grande
                title=dict(font=dict(size=24)),  # Título más grande
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )
            
            fig.write_html(f'visualizaciones/{nombre}_tendencia_anual.html')
            
            # Si hay una columna de categoría, analizar tendencias por categoría
            if columna_categoria and columna_categoria in df.columns:
                if df[columna_categoria].nunique() <= 10:  # Solo si hay un número razonable de categorías
                    # Crear copia del dataframe para no modificar el original
                    df_copy = df.copy()
                    
                    # Normalizar valores de género si corresponde
                    if columna_categoria.upper() in ['GENERO', 'GÉNERO', 'GÃ©NERO']:
                        # Normalizar valores no reportados
                        no_reportado = ['NO REPORTA', '-', 'NO REPORTADO', 'NO INFORMA', 'SIN INFORMACIÓN', 'DESCONOCIDO']
                        df_copy[columna_categoria] = df_copy[columna_categoria].replace(no_reportado, 'NO REPORTADO')
                    
                    category_yearly = df_copy.groupby(['año', columna_categoria]).size().reset_index(name='cantidad')
                    
                    fig = px.line(category_yearly, x='año', y='cantidad', color=columna_categoria,
                                 title=f'Tendencia Anual por {columna_categoria} - {nombre}',
                                 labels={'cantidad': 'Cantidad de Casos', 'año': 'Año'},
                                 markers=True,
                                 width=1200,  # Mayor ancho
                                 height=700)  # Mayor altura
                    
                    fig.update_layout(
                        template='plotly_white',
                        legend_title_text=columna_categoria,
                        plot_bgcolor='white',
                        font=dict(family="Arial", size=14),  # Fuente más grande
                        title=dict(font=dict(size=24)),  # Título más grande
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray')
                    )
                    
                    fig.write_html(f'visualizaciones/{nombre}_tendencia_por_{columna_categoria}.html')
            
            return True
        else:
            print(f"  No se encontraron columnas de fecha válidas en {nombre}")
            return False
    except Exception as e:
        print(f"  Error al analizar tendencias temporales en {nombre}: {e}")
        return False

# Analizar tendencias en cada dataset según sus columnas
for nombre, df in datasets.items():
    fecha_encontrada = False
    
    # Caso 1: Columnas separadas de Año, Mes, Día (con normalización de nombres y manejo de codificación)
    # Buscar todas las columnas que podrían ser año, mes o día
    posibles_cols_anio = [col for col in df.columns if 'AÃ±o' in col or 'Año' in col or 'ANO' in col.upper() or 'YEAR' in col.upper()]
    posibles_cols_mes = [col for col in df.columns if 'Mes' in col or 'MES' in col.upper() or 'MONTH' in col.upper()]
    posibles_cols_dia = [col for col in df.columns if 'DÃ­a' in col or 'Día' in col or 'DIA' in col.upper() or 'DAY' in col.upper()]
    
    print(f"\nAnalizando tendencias temporales en {nombre}...")
    print(f"  Columnas de año encontradas: {posibles_cols_anio}")
    print(f"  Columnas de mes encontradas: {posibles_cols_mes}")
    print(f"  Columnas de día encontradas: {posibles_cols_dia}")
    
    if posibles_cols_anio:
        print(f"  Usando columna de año: {posibles_cols_anio[0]}")
        
        # Identificar posibles columnas de categoría según el dataset
        col_categoria = None
        for posible_col in ['Armas / Medios', 'MODALIDAD', 'DELITO', 'TIPO', 'GÃ©nero', 'Género', 'GENERO']:
            if posible_col in df.columns:
                col_categoria = posible_col
                break
        
        # Usar el valor de año directamente para crear series de tiempo
        anio_col = posibles_cols_anio[0]
        df['año_num'] = pd.to_numeric(df[anio_col], errors='coerce')
        
        # Crear agregación por año
        yearly_counts = df.groupby('año_num').size().reset_index(name='cantidad')
        
        # Visualización con Plotly
        fig = px.line(yearly_counts, x='año_num', y='cantidad', 
                     title=f'Tendencia Anual - {nombre}',
                     labels={'cantidad': 'Cantidad de Casos', 'año_num': 'Año'},
                     markers=True,
                     width=1200,  # Mayor ancho
                     height=700)  # Mayor altura
        
        fig.update_layout(
            template='plotly_white',
            legend_title_text='',
            plot_bgcolor='white',
            font=dict(family="Arial", size=14),  # Fuente más grande
            title=dict(font=dict(size=24)),  # Título más grande
            xaxis=dict(showgrid=True, gridcolor='lightgray'),
            yaxis=dict(showgrid=True, gridcolor='lightgray')
        )
        
        fig.write_html(f'visualizaciones/{nombre}_tendencia_anual.html')
        
        # Si hay una columna de categoría, analizar tendencias por categoría
        if col_categoria and col_categoria in df.columns:
            if df[col_categoria].nunique() <= 10:  # Solo si hay un número razonable de categorías
                # Crear copia del dataframe para no modificar el original
                df_copy = df.copy()
                
                # Normalizar valores de género si corresponde
                if col_categoria.upper() in ['GENERO', 'GÉNERO', 'GÃ©NERO']:
                    # Normalizar valores no reportados
                    no_reportado = ['NO REPORTA', '-', 'NO REPORTADO', 'NO INFORMA', 'SIN INFORMACIÓN', 'DESCONOCIDO']
                    df_copy[col_categoria] = df_copy[col_categoria].replace(no_reportado, 'NO REPORTADO')
                
                category_yearly = df_copy.groupby(['año_num', col_categoria]).size().reset_index(name='cantidad')
                
                fig = px.line(category_yearly, x='año_num', y='cantidad', color=col_categoria,
                             title=f'Tendencia Anual por {col_categoria} - {nombre}',
                             labels={'cantidad': 'Cantidad de Casos', 'año_num': 'Año'},
                             markers=True,
                             width=1200,  # Mayor ancho
                             height=700)  # Mayor altura
                
                fig.update_layout(
                    template='plotly_white',
                    legend_title_text=col_categoria,
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=14),  # Fuente más grande
                    title=dict(font=dict(size=24)),  # Título más grande
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                
                fig.write_html(f'visualizaciones/{nombre}_tendencia_por_{col_categoria}.html')
        
        fecha_encontrada = True
    
    # Caso 2: Columna única de fecha
    if not fecha_encontrada:
        # Intentar encontrar la columna de fecha correcta (varios nombres posibles)
        columnas_fecha_posibles = [
            'FECHA', 'FECHA_HECHO', 'FECHA HECHO', 'FECHA_COMISION', 'FECHA COMISION',
            'FECHA_REGISTRO', 'FECHA REGISTRO'
        ]
        
        for col_fecha in columnas_fecha_posibles:
            if col_fecha in df.columns:
                # Identificar posibles columnas de categoría según el dataset
                col_categoria = None
                if 'MODALIDAD' in df.columns:
                    col_categoria = 'MODALIDAD'
                elif 'DELITO' in df.columns:
                    col_categoria = 'DELITO'
                elif 'TIPO' in df.columns:
                    col_categoria = 'TIPO'
                elif 'GENERO' in df.columns:
                    col_categoria = 'GENERO'
                elif 'ARMAS MEDIOS' in df.columns:
                    col_categoria = 'ARMAS MEDIOS'
                elif 'Armas / Medios' in df.columns:
                    col_categoria = 'Armas / Medios'
                    
                fecha_encontrada = analizar_tendencia_temporal(df, nombre, col_fecha, col_categoria)
                break
    
    if not fecha_encontrada:
        print(f"  No se pudo encontrar una columna de fecha válida en {nombre}")

# ==========================================
# Análisis geográfico
# ==========================================

def analizar_geografia(df, nombre, col_departamento='DEPARTAMENTO', col_municipio='MUNICIPIO'):
    """Analiza y visualiza la distribución geográfica de los casos"""
    try:
        print(f"\nAnalizando distribución geográfica en {nombre}...")
        
        # Verificar si existen las columnas geográficas
        geo_cols = []
        if col_departamento in df.columns:
            geo_cols.append(col_departamento)
        if col_municipio in df.columns:
            geo_cols.append(col_municipio)
            
        if len(geo_cols) > 0:
            # Analizar por departamento
            if col_departamento in df.columns:
                dept_counts = df[col_departamento].value_counts().reset_index()
                dept_counts.columns = [col_departamento, 'cantidad']
                dept_counts = dept_counts.sort_values('cantidad', ascending=False).head(15)
                
                fig = px.bar(dept_counts, y=col_departamento, x='cantidad', 
                            title=f'Departamentos con Mayor Incidencia - {nombre}',
                            labels={'cantidad': 'Cantidad de Casos', col_departamento: 'Departamento'},
                            color='cantidad', 
                            color_continuous_scale='Viridis',
                            orientation='h',
                            width=1200,  # Mayor ancho
                            height=800)  # Mayor altura
                
                fig.update_layout(
                    template='plotly_white',
                    legend_title_text='',
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=14),  # Fuente más grande
                    title=dict(font=dict(size=24)),  # Título más grande
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(categoryorder='total ascending')
                )
                
                fig.write_html(f'visualizaciones/{nombre}_distribucion_departamentos.html')
                
            return True
        else:
            print(f"  No se encontraron columnas geográficas en {nombre}")
            return False
    except Exception as e:
        print(f"  Error al analizar distribución geográfica en {nombre}: {e}")
        return False

# Analizar geografía en cada dataset
for nombre, df in datasets.items():
    # Diferentes posibles nombres para columnas geográficas
    cols_departamento = ['DEPARTAMENTO', 'DEPTO', 'DEPARTAMENTO_HECHO', 'DEPTO_HECHO']
    cols_municipio = ['MUNICIPIO', 'CIUDAD', 'MUNICIPIO_HECHO', 'CIUDAD_HECHO']
    
    for col_depto in cols_departamento:
        for col_muni in cols_municipio:
            if col_depto in df.columns or col_muni in df.columns:
                analizar_geografia(df, nombre, col_depto, col_muni)
                break

# ==========================================
# Análisis de variables categóricas
# ==========================================

def analizar_variables_categoricas(df, nombre):
    """Analiza y visualiza las variables categóricas más importantes"""
    print(f"\nAnalizando variables categóricas en {nombre}...")
    
    # Lista de posibles columnas categóricas de interés según el dataset
    columnas_categoricas = []
    
    # Definir columnas específicas por dataset
    if nombre == 'Capturas':
        columnas_categoricas = ['DESCRIPCION CONDUCTA CAPTURA', 'GENERO']
    elif nombre == 'Delitos_Contra_Medio_Ambiente':
        columnas_categoricas = ['DESCRIPCION_CONDUCTA', 'MUNICIPIO', 'ZONA']
    elif nombre == 'Delitos_Informáticos':
        columnas_categoricas = ['Descripcion Conducta', 'Municipio']
    elif nombre == 'Homicidios':
        columnas_categoricas = ['Clase de Sitio', 'Armas / Medios', 'GÃ©nero', 'Género', 'Zona']
    elif nombre == 'Hurto_Automotores':
        columnas_categoricas = ['Armas / Medios', 'Zona', 'Clase de Sitio', 'Clase Bien']
    elif nombre == 'Hurto_Comercio':
        columnas_categoricas = ['Armas / Medios', 'Zona', 'Clase de Sitio']
    elif nombre == 'Hurto_Personas':
        columnas_categoricas = ['Armas / Medios', 'GÃ©nero', 'Género', 'Zona', 'Clase de Sitio']
    elif nombre == 'Incautación_Estupefacientes':
        columnas_categoricas = ['CLASE BIEN', 'MUNICIPIO']
    elif nombre == 'invasión_Usurpación_Tierras':
        columnas_categoricas = ['DESCRIPCION CONDUCTA', 'MUNICIPIO']
    elif nombre == 'Presupuesto_de_Gastos':
        columnas_categoricas = ['CONCEPTO']
    elif nombre == 'Violencia_Intrafamiliar':
        columnas_categoricas = ['MUNICIPIO', 'ARMAS MEDIOS', 'GENERO']
    elif nombre == 'Frentes_De_Seguridad':
        columnas_categoricas = ['ESTADO', 'ZONA']
    else:
        # Lista genérica si no se encuentra el dataset específico
        columnas_categoricas = [
            'MODALIDAD', 'TIPO', 'ARMAS MEDIOS', 'Armas / Medios', 'GENERO', 'GÃ©nero', 'Género', 'DELITO', 
            'TIPO_HURTO', 'TIPO_HOMICIDIO', 'SEXO', 'CLASE', 'MARCA', 'COLOR', 'ZONA', 'Zona',
            'DESCRIPCION CONDUCTA', 'DESCRIPCION CONDUCTA CAPTURA', 'DESCRIPCION_CONDUCTA',
            'Clase de Sitio', 'CLASE BIEN', 'Descripcion Conducta', 'MUNICIPIO', 'Municipio'
        ]
    
    visualizaciones_creadas = 0
    
    # Caso especial para Presupuesto_de_Gastos con columnas numéricas
    if nombre == 'Presupuesto_de_Gastos' and 'CONCEPTO' in df.columns:
        try:
            # Convertir columnas numéricas: quitar comas y convertir a float
            for col in ['PRESUPUESTO VIGENTE (PV)', 'COMPROMISOS (CP)', 'PAGOS (PG)']:
                if col in df.columns:
                    df[col] = df[col].replace({',': ''}, regex=True).astype(float)
            
            # Agrupar por concepto y sumar valores
            concepto_presupuesto = df.groupby('CONCEPTO')['PRESUPUESTO VIGENTE (PV)'].sum().reset_index()
            concepto_compromisos = df.groupby('CONCEPTO')['COMPROMISOS (CP)'].sum().reset_index()
            concepto_pagos = df.groupby('CONCEPTO')['PAGOS (PG)'].sum().reset_index()
            
            # Ordenar por presupuesto
            concepto_presupuesto = concepto_presupuesto.sort_values('PRESUPUESTO VIGENTE (PV)', ascending=False)
            
            # Convertir a formato largo para Plotly
            concepto_top = concepto_presupuesto.head(15)['CONCEPTO'].tolist()
            
            df_filtered = df[df['CONCEPTO'].isin(concepto_top)].copy()
            
            # Preparar datos en formato largo
            presupuesto_largo = pd.melt(
                df_filtered, 
                id_vars=['CONCEPTO'], 
                value_vars=['PRESUPUESTO VIGENTE (PV)', 'COMPROMISOS (CP)', 'PAGOS (PG)'],
                var_name='Tipo', 
                value_name='Valor'
            )
            
            # Calcular totales por concepto y tipo
            presupuesto_agg = presupuesto_largo.groupby(['CONCEPTO', 'Tipo'])['Valor'].sum().reset_index()
            
            # Crear visualización
            fig = px.bar(
                presupuesto_agg, 
                x='CONCEPTO', 
                y='Valor', 
                color='Tipo',
                title=f'Presupuesto, Compromisos y Pagos por Concepto - {nombre}',
                labels={'Valor': 'Monto (COP)', 'CONCEPTO': 'Concepto', 'Tipo': 'Tipo de Valor'},
                height=900,  # Mayor altura para mejor visibilidad
                width=1200   # Mayor ancho
            )
            
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                font=dict(family="Arial", size=14),  # Fuente más grande
                title=dict(font=dict(size=24)),  # Título más grande
                xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=45),
                yaxis=dict(showgrid=True, gridcolor='lightgray'),
                legend_title_text='Tipo de Valor'
            )
            
            # Guardar visualización
            nombre_archivo = f"visualizaciones/{nombre}_presupuesto_concepto.html"
            fig.write_html(nombre_archivo)
            print(f"  Visualización creada: {nombre_archivo}")
            
            visualizaciones_creadas += 1
            
        except Exception as e:
            print(f"  Error al analizar presupuesto en {nombre}: {e}")
    
    # Analizar columnas categóricas generales
    for col in columnas_categoricas:
        if col in df.columns:
            try:
                # Crear copia para no modificar el original
                df_copy = df.copy()
                
                # Normalizar valores de género
                if col.upper() in ['GENERO', 'GÉNERO', 'GÃ©NERO']:
                    # Normalizar valores no reportados
                    no_reportado = ['NO REPORTA', '-', 'NO REPORTADO', 'NO INFORMA', 'SIN INFORMACIÓN', 'DESCONOCIDO']
                    df_copy[col] = df_copy[col].replace(no_reportado, 'NO REPORTADO')
                
                # Contar valores
                conteo = df_copy[col].value_counts().reset_index()
                conteo.columns = [col, 'cantidad']
                
                # Manejar datasets con muchos valores (como municipios)
                if col.upper() in ['MUNICIPIO', 'DESCRIPCION CONDUCTA', 'DESCRIPCION CONDUCTA CAPTURA', 'DESCRIPCION_CONDUCTA']:
                    # Tomar solo los más frecuentes para la visualización
                    if len(conteo) > 25:
                        conteo = conteo.head(25)
                else:
                    # Para otras columnas, tomar top 15
                    if len(conteo) > 15:
                        conteo = conteo.head(15)
                
                # Crear visualización
                fig = px.bar(conteo, y=col, x='cantidad',
                            title=f'Distribución por {col} - {nombre}',
                            labels={'cantidad': 'Cantidad de Casos', col: col},
                            color='cantidad',
                            color_continuous_scale='Viridis',
                            orientation='h',
                            height=max(700, len(conteo) * 40),  # Mayor altura según cantidad de categorías
                            width=1200)  # Hacer más ancho el gráfico
                
                fig.update_layout(
                    template='plotly_white',
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=14),  # Fuente más grande
                    title=dict(font=dict(size=24)),  # Título más grande
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(categoryorder='total ascending', tickfont=dict(size=14))  # Aumentar tamaño de fuente
                )
                
                # Guardar visualización
                column_name = col.replace(' ', '_').replace('/', '_').lower()
                nombre_archivo = f"visualizaciones/{nombre}_{column_name}.html"
                fig.write_html(nombre_archivo)
                print(f"  Visualización creada: {nombre_archivo}")
                
                visualizaciones_creadas += 1
                
                # Si también hay columna de año, analizar evolución de categorías en el tiempo
                posibles_cols_anio = [c for c in df.columns if 'AÃ±o' in c or 'Año' in c or 'ANO' in c.upper() or 'YEAR' in c.upper()]
                
                if posibles_cols_anio:
                    anio_col = posibles_cols_anio[0]
                    
                    # Convertir a numérico
                    df_copy['año_num'] = pd.to_numeric(df_copy[anio_col], errors='coerce')
                    
                    # Obtener las 5 categorías más frecuentes
                    top_categorias = conteo.head(5)[col].tolist()
                    
                    # Filtrar datos
                    df_filtrado = df_copy[df_copy[col].isin(top_categorias)]
                    
                    # Agrupar por año y categoría
                    evolucion = df_filtrado.groupby(['año_num', col]).size().reset_index(name='cantidad')
                    
                    # Crear visualización
                    fig = px.line(evolucion, x='año_num', y='cantidad', color=col,
                                 title=f'Evolución de principales {col} - {nombre}',
                                 labels={'cantidad': 'Cantidad de Casos', 'año_num': 'Año', col: col},
                                 markers=True,
                                 height=700,
                                 width=1200)
                    
                    fig.update_layout(
                        template='plotly_white',
                        plot_bgcolor='white',
                        font=dict(family="Arial", size=14),  # Fuente más grande
                        title=dict(font=dict(size=24)),  # Título más grande
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(showgrid=True, gridcolor='lightgray'),
                        legend_title_text=col
                    )
                    
                    # Guardar visualización
                    nombre_archivo = f"visualizaciones/{nombre}_{column_name}_evolucion.html"
                    fig.write_html(nombre_archivo)
                    print(f"  Visualización creada: {nombre_archivo}")
                    
                    visualizaciones_creadas += 1
                
            except Exception as e:
                print(f"  Error al analizar {col} en {nombre}: {e}")
    
    return visualizaciones_creadas > 0

# ==========================================
# Análisis de correlaciones entre datasets
# ==========================================

print("\nAnalizando posibles correlaciones entre datasets...")

# Intentar encontrar correlaciones entre homicidios y capturas
if 'Homicidios' in datasets and 'Capturas' in datasets:
    try:
        # Preparar datos por año
        homicidios = datasets['Homicidios']
        capturas = datasets['Capturas']
        
        if 'FECHA' in homicidios.columns and 'FECHA' in capturas.columns:
            # Convertir fechas
            homicidios['FECHA'] = pd.to_datetime(homicidios['FECHA'], errors='coerce')
            capturas['FECHA'] = pd.to_datetime(capturas['FECHA'], errors='coerce')
            
            # Agregar por año
            homicidios['año'] = homicidios['FECHA'].dt.year
            capturas['año'] = capturas['FECHA'].dt.year
            
            hom_por_año = homicidios.groupby('año').size().reset_index(name='homicidios')
            cap_por_año = capturas.groupby('año').size().reset_index(name='capturas')
            
            # Unir datos
            correlacion = pd.merge(hom_por_año, cap_por_año, on='año', how='inner')
            
            if not correlacion.empty:
                # Crear visualización
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Scatter(x=correlacion['año'], y=correlacion['homicidios'], 
                              name="Homicidios", line=dict(color='red', width=3)),
                    secondary_y=False
                )
                
                fig.add_trace(
                    go.Scatter(x=correlacion['año'], y=correlacion['capturas'], 
                              name="Capturas", line=dict(color='blue', width=3)),
                    secondary_y=True
                )
                
                fig.update_layout(
                    title='Relación entre Homicidios y Capturas por Año',
                    template='plotly_white',
                    font=dict(family="Arial", size=14),  # Fuente más grande
                    title_font=dict(size=24),  # Título más grande
                    plot_bgcolor='white',
                    legend_title_text='',
                    xaxis=dict(title="Año", showgrid=True, gridcolor='lightgray'),
                    width=1200,  # Mayor ancho
                    height=700,  # Mayor altura
                )
                
                fig.update_yaxes(title_text="Número de Homicidios", secondary_y=False)
                fig.update_yaxes(title_text="Número de Capturas", secondary_y=True)
                
                fig.write_html('visualizaciones/correlacion_homicidios_capturas.html')
                
                # Calcular correlación
                corr = correlacion['homicidios'].corr(correlacion['capturas'])
                print(f"  Correlación entre homicidios y capturas: {corr:.2f}")
    except Exception as e:
        print(f"  Error al analizar correlación entre homicidios y capturas: {e}")

# ==========================================
# Análisis de variables categóricas
# ==========================================

print("\nAnalizando variables categóricas en los datasets...")
for nombre, df in datasets.items():
    analizar_variables_categoricas(df, nombre)

print("\nAnálisis completado. Visualizaciones guardadas en la carpeta 'visualizaciones'.") 