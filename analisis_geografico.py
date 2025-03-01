import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import folium
from folium.plugins import HeatMap, MarkerCluster
import warnings
warnings.filterwarnings('ignore')

# Set plot styles for better aesthetics
plt.style.use('ggplot')
sns.set_style('whitegrid')
sns.set_context('talk')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

# Create output directory for visualizations
os.makedirs('visualizaciones', exist_ok=True)

print("Iniciando análisis geográfico de seguridad y criminalidad...")

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
        
        print(f"  Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
        return df
    except Exception as e:
        # Si falla, intentar con coma como delimitador
        try:
            df = pd.read_csv(f"datos/{filename}", encoding='latin1', low_memory=False)
            print(f"  Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
            return df
        except Exception as e:
            print(f"Error al cargar {filename}: {e}")
            return None

# ==========================================
# Análisis geográfico detallado
# ==========================================

def analizar_distribucion_geografica(df, nombre, col_lat='LATITUD', col_lon='LONGITUD', 
                                    col_depto='DEPARTAMENTO', col_muni='MUNICIPIO'):
    """Analiza y visualiza la distribución geográfica de los datos con mapas"""
    try:
        print(f"\nAnalizando distribución geográfica en {nombre}...")
        
        # Análisis por departamento
        if col_depto in df.columns:
            top_deptos = df[col_depto].value_counts().head(10).reset_index()
            top_deptos.columns = [col_depto, 'cantidad']
            
            fig = px.bar(top_deptos, y=col_depto, x='cantidad',
                        title=f'Top 10 Departamentos - {nombre}',
                        labels={'cantidad': 'Cantidad de Casos', col_depto: 'Departamento'},
                        color='cantidad',
                        color_continuous_scale='Viridis',
                        orientation='h')
            
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                font=dict(family="Arial", size=12),
                title=dict(font=dict(size=20)),
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(categoryorder='total ascending')
            )
            
            fig.write_html(f'visualizaciones/{nombre}_top_departamentos.html')
            
            # Análisis por municipio si está disponible
            if col_muni in df.columns:
                # Obtener top municipios
                top_munis = df[col_muni].value_counts().head(15).reset_index()
                top_munis.columns = [col_muni, 'cantidad']
                
                fig = px.bar(top_munis, y=col_muni, x='cantidad',
                            title=f'Top 15 Municipios - {nombre}',
                            labels={'cantidad': 'Cantidad de Casos', col_muni: 'Municipio'},
                            color='cantidad',
                            color_continuous_scale='Viridis',
                            orientation='h')
                
                fig.update_layout(
                    template='plotly_white',
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=12),
                    title=dict(font=dict(size=20)),
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(categoryorder='total ascending')
                )
                
                fig.write_html(f'visualizaciones/{nombre}_top_municipios.html')
                
                # Análisis cruzado departamento vs municipio
                # Primero filtrar para top 5 departamentos
                top5_deptos = df[col_depto].value_counts().head(5).index.tolist()
                df_filtrado = df[df[col_depto].isin(top5_deptos)]
                
                # Agrupar por departamento y municipio
                grouped = df_filtrado.groupby([col_depto, col_muni]).size().reset_index(name='cantidad')
                
                # Ordenar por cantidad total y tomar los top 20
                grouped = grouped.sort_values('cantidad', ascending=False).head(20)
                
                fig = px.bar(grouped, x=col_muni, y='cantidad', color=col_depto,
                            title=f'Top 20 Combinaciones Departamento-Municipio - {nombre}',
                            labels={'cantidad': 'Cantidad de Casos', 
                                   col_muni: 'Municipio', 
                                   col_depto: 'Departamento'},
                            barmode='group')
                
                fig.update_layout(
                    template='plotly_white',
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=12),
                    title=dict(font=dict(size=20)),
                    xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=45),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                
                fig.write_html(f'visualizaciones/{nombre}_depto_municipio.html')
        
        # Crear mapa si hay coordenadas disponibles
        if col_lat in df.columns and col_lon in df.columns:
            # Filtrar registros con coordenadas válidas
            df_coords = df.copy()
            
            # Convertir a numérico y filtrar valores nulos y cero
            df_coords[col_lat] = pd.to_numeric(df_coords[col_lat], errors='coerce')
            df_coords[col_lon] = pd.to_numeric(df_coords[col_lon], errors='coerce')
            
            # Filtrar valores en Colombia (aproximadamente)
            df_coords = df_coords[
                (df_coords[col_lat].between(-4.2, 13.0)) & 
                (df_coords[col_lon].between(-82.0, -66.0))
            ]
            
            if not df_coords.empty and len(df_coords) > 10:  # Solo si hay suficientes puntos
                print(f"  Creando mapa con {len(df_coords)} puntos georreferenciados")
                
                # Crear mapa base centrado en Colombia
                mapa = folium.Map(
                    location=[4.570868, -74.297333],  # Coordenadas aproximadas de Colombia
                    zoom_start=6,
                    tiles='CartoDB positron'
                )
                
                # Limitar a máximo 5000 puntos para rendimiento
                if len(df_coords) > 5000:
                    df_coords = df_coords.sample(5000, random_state=42)
                
                # Agregar puntos al mapa usando clusters
                marker_cluster = MarkerCluster().add_to(mapa)
                
                for idx, row in df_coords.iterrows():
                    popup_text = f"{nombre}<br>"
                    
                    # Agregar información adicional al popup si está disponible
                    if col_depto in df.columns and not pd.isna(row[col_depto]):
                        popup_text += f"Departamento: {row[col_depto]}<br>"
                    if col_muni in df.columns and not pd.isna(row[col_muni]):
                        popup_text += f"Municipio: {row[col_muni]}<br>"
                    
                    # Agregar fecha si está disponible
                    if 'FECHA' in df.columns and not pd.isna(row['FECHA']):
                        popup_text += f"Fecha: {row['FECHA']}<br>"
                    
                    folium.Marker(
                        [row[col_lat], row[col_lon]],
                        popup=folium.Popup(popup_text, max_width=300)
                    ).add_to(marker_cluster)
                
                # Crear mapa de calor
                heat_data = [[row[col_lat], row[col_lon]] for idx, row in df_coords.iterrows()]
                HeatMap(heat_data, radius=15).add_to(mapa)
                
                # Guardar mapa
                mapa.save(f'visualizaciones/{nombre}_mapa.html')
        
        return True
    except Exception as e:
        print(f"  Error al analizar distribución geográfica en {nombre}: {e}")
        return False

# ==========================================
# Análisis de frentes de seguridad
# ==========================================

def analizar_frentes_seguridad():
    """Analiza los frentes de seguridad y su relación con incidentes"""
    print("\nAnalizando frentes de seguridad...")
    
    try:
        # Cargar dataset de frentes de seguridad
        df_frentes = load_dataset("Frentes_De_Seguridad.csv")
        
        if df_frentes is not None:
            print("  Columnas disponibles:", df_frentes.columns.tolist())
            
            # Identificar columnas por posición en lugar de nombres debido a problemas de codificación
            # Usualmente, la estructura es:
            # 0: REGIÓN, 1: METROPOLITANA, 2: DISTRITO, 3: ESTACIÓN, 4: BARRIO, 5: ZONA, 6: NRO INTEGRANTES, 7: ESTADO
            
            # Normalizar nombres de columnas
            new_cols = ['REGION', 'METROPOLITANA', 'DISTRITO', 'ESTACION', 'BARRIO', 'ZONA', 'NRO_INTEGRANTES', 'ESTADO']
            if len(df_frentes.columns) == 8:  # Si tiene las 8 columnas esperadas
                df_frentes.columns = new_cols
                print("  Normalizados los nombres de columnas")
            
            # Extraer localidad de la última palabra de la columna ESTACION
            if 'ESTACION' in df_frentes.columns:
                print("  Extrayendo localidad desde la columna ESTACION...")
                # Extraer la última palabra de ESTACIÓN como localidad
                df_frentes['LOCALIDAD'] = df_frentes['ESTACION'].str.split().str[-1]
                
                # Análisis por localidad
                localidad_counts = df_frentes['LOCALIDAD'].value_counts().reset_index()
                localidad_counts.columns = ['LOCALIDAD', 'cantidad']
                
                # Crear visualización de localidades
                fig = px.bar(localidad_counts.head(15), y='LOCALIDAD', x='cantidad',
                            title='Distribución de Frentes de Seguridad por Localidad',
                            labels={'cantidad': 'Cantidad de Frentes', 'LOCALIDAD': 'Localidad'},
                            color='cantidad',
                            color_continuous_scale='Viridis',
                            orientation='h')
                
                fig.update_layout(
                    template='plotly_white',
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=12),
                    title=dict(font=dict(size=20)),
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(categoryorder='total ascending')
                )
                
                fig.write_html('visualizaciones/frentes_seguridad_localidades.html')
                
                # Si también existe columna BARRIO, analizar por barrio
                if 'BARRIO' in df_frentes.columns:
                    print("  Analizando distribución por barrio...")
                    barrio_counts = df_frentes['BARRIO'].value_counts().reset_index()
                    barrio_counts.columns = ['BARRIO', 'cantidad']
                    
                    # Crear visualización de barrios (top 20)
                    fig = px.bar(barrio_counts.head(20), y='BARRIO', x='cantidad',
                                title='Top 20 Barrios con Frentes de Seguridad',
                                labels={'cantidad': 'Cantidad de Frentes', 'BARRIO': 'Barrio'},
                                color='cantidad',
                                color_continuous_scale='Viridis',
                                orientation='h')
                    
                    fig.update_layout(
                        template='plotly_white',
                        plot_bgcolor='white',
                        font=dict(family="Arial", size=12),
                        title=dict(font=dict(size=20)),
                        xaxis=dict(showgrid=True, gridcolor='lightgray'),
                        yaxis=dict(categoryorder='total ascending')
                    )
                    
                    fig.write_html('visualizaciones/frentes_seguridad_barrios.html')
                    
                    # Análisis cruzado de localidad y barrio
                    print("  Generando análisis cruzado de localidad y barrio...")
                    # Seleccionar top 5 localidades
                    top_localidades = localidad_counts.head(5)['LOCALIDAD'].tolist()
                    df_top_loc = df_frentes[df_frentes['LOCALIDAD'].isin(top_localidades)]
                    
                    # Para cada localidad, encontrar los barrios más frecuentes
                    loc_barrio_data = []
                    
                    for localidad in top_localidades:
                        df_loc = df_top_loc[df_top_loc['LOCALIDAD'] == localidad]
                        top_barrios = df_loc['BARRIO'].value_counts().head(5)
                        
                        for barrio, count in top_barrios.items():
                            loc_barrio_data.append({
                                'LOCALIDAD': localidad,
                                'BARRIO': barrio,
                                'cantidad': count
                            })
                    
                    df_loc_barrio = pd.DataFrame(loc_barrio_data)
                    
                    # Crear visualización de barrios por localidad
                    if not df_loc_barrio.empty:
                        fig = px.bar(df_loc_barrio, x='BARRIO', y='cantidad', color='LOCALIDAD',
                                    title='Principales Barrios por Localidad con Frentes de Seguridad',
                                    labels={'cantidad': 'Cantidad de Frentes', 'BARRIO': 'Barrio', 'LOCALIDAD': 'Localidad'},
                                    barmode='group')
                        
                        fig.update_layout(
                            template='plotly_white',
                            plot_bgcolor='white',
                            font=dict(family="Arial", size=12),
                            title=dict(font=dict(size=20)),
                            xaxis=dict(showgrid=True, gridcolor='lightgray', tickangle=45),
                            yaxis=dict(showgrid=True, gridcolor='lightgray')
                        )
                        
                        fig.write_html('visualizaciones/frentes_seguridad_localidad_barrio.html')
                
                # Crear mapa de calor si tenemos suficientes datos
                if 'LOCALIDAD' in df_frentes.columns and 'BARRIO' in df_frentes.columns:
                    print("  Generando mapa de calor de frentes por localidad y tipo...")
                    
                    # Crear matriz de localidad vs estado
                    if 'ESTADO' in df_frentes.columns:
                        # Pivot table de localidad vs estado
                        pivot = pd.crosstab(df_frentes['LOCALIDAD'], df_frentes['ESTADO'])
                        
                        # Seleccionar las 15 localidades con más frentes
                        top_localidades = df_frentes['LOCALIDAD'].value_counts().head(15).index
                        pivot_filtered = pivot.loc[pivot.index.isin(top_localidades)]
                        
                        # Crear heatmap
                        fig = px.imshow(pivot_filtered,
                                       labels=dict(x="Estado del Frente", y="Localidad", color="Cantidad"),
                                       title='Distribución de Estado de Frentes por Localidad',
                                       color_continuous_scale='Viridis')
                        
                        fig.update_layout(
                            template='plotly_white',
                            font=dict(family="Arial", size=12),
                            title=dict(font=dict(size=20)),
                            xaxis=dict(tickangle=45)
                        )
                        
                        fig.write_html('visualizaciones/frentes_seguridad_heatmap_estado_localidad.html')
            
            # Análisis por zona y número de integrantes
            if 'NRO_INTEGRANTES' in df_frentes.columns and 'ZONA' in df_frentes.columns:
                print("  Analizando distribución por número de integrantes y zona...")
                # Convertir a numérico el número de integrantes
                df_frentes['NRO_INTEGRANTES'] = pd.to_numeric(df_frentes['NRO_INTEGRANTES'], errors='coerce')
                
                # Agrupar por zona y calcular estadísticas
                zona_stats = df_frentes.groupby('ZONA')['NRO_INTEGRANTES'].agg(['count', 'mean', 'std', 'min', 'max']).reset_index()
                zona_stats.columns = ['ZONA', 'Cantidad', 'Promedio', 'Desviación', 'Mínimo', 'Máximo']
                
                # Visualización de número de integrantes por zona
                fig = px.bar(zona_stats, x='ZONA', y='Promedio', 
                            error_y='Desviación',
                            title='Promedio de Integrantes por Zona',
                            labels={'Promedio': 'Promedio de Integrantes', 'ZONA': 'Zona'},
                            color='Cantidad',
                            color_continuous_scale='Viridis',
                            text='Cantidad')
                
                fig.update_layout(
                    template='plotly_white',
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=12),
                    title=dict(font=dict(size=20)),
                    xaxis=dict(showgrid=True, gridcolor='lightgray'),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                
                fig.write_html('visualizaciones/frentes_seguridad_integrantes_zona.html')
            
            # Análisis por estado
            if 'ESTADO' in df_frentes.columns:
                estado_counts = df_frentes['ESTADO'].value_counts().reset_index()
                estado_counts.columns = ['ESTADO', 'cantidad']
                
                fig = px.pie(estado_counts, values='cantidad', names='ESTADO',
                            title='Distribución de Frentes de Seguridad por Estado',
                            color_discrete_sequence=px.colors.qualitative.Bold)
                
                fig.update_layout(
                    template='plotly_white',
                    font=dict(family="Arial", size=12),
                    title=dict(font=dict(size=20))
                )
                
                fig.write_html('visualizaciones/frentes_seguridad_estados.html')
            
            return True
    except Exception as e:
        print(f"  Error al analizar frentes de seguridad: {e}")
    
    return False

# ==========================================
# Análisis cruzado de zonas de delitos
# ==========================================

def analizar_zonas_delitos():
    """Analiza y compara las zonas geográficas de diferentes tipos de delitos"""
    print("\nAnalizando distribución geográfica de diferentes delitos...")
    
    try:
        # Tipos de delitos a comparar
        tipos_delitos = {
            'Homicidios': 'Homicidios.csv',
            'Hurto a Personas': 'Hurto_Personas.csv',
            'Hurto a Comercio': 'Hurto_Comercio.csv'
        }
        
        # Estructura para almacenar datos departamentales
        datos_por_depto = {}
        
        for tipo, archivo in tipos_delitos.items():
            df = load_dataset(archivo)
            
            if df is not None:
                # Buscar columna de departamento
                col_depto = next((col for col in df.columns if 'DEPART' in col.upper() or 'DEPTO' in col.upper()), None)
                
                if col_depto:
                    # Contar por departamento
                    conteo = df[col_depto].value_counts().reset_index()
                    conteo.columns = [col_depto, 'cantidad']
                    
                    # Solo los top 10 departamentos
                    conteo = conteo.head(10)
                    
                    # Almacenar datos
                    datos_por_depto[tipo] = {
                        'columna': col_depto,
                        'data': conteo
                    }
        
        # Si hay datos para al menos dos tipos de delitos, crear visualización comparativa
        if len(datos_por_depto) >= 2:
            # Crear figura con subplots
            fig = make_subplots(rows=len(datos_por_depto), cols=1, 
                               subplot_titles=[f"Top 10 Departamentos - {tipo}" for tipo in datos_por_depto.keys()],
                               vertical_spacing=0.1)
            
            # Agregar datos para cada tipo de delito
            for i, (tipo, datos) in enumerate(datos_por_depto.items(), 1):
                col_depto = datos['columna']
                conteo = datos['data']
                
                fig.add_trace(
                    go.Bar(
                        y=conteo[col_depto],
                        x=conteo['cantidad'],
                        name=tipo,
                        orientation='h',
                        marker=dict(color=px.colors.qualitative.Plotly[i-1])
                    ),
                    row=i, col=1
                )
            
            # Actualizar layout
            fig.update_layout(
                title='Comparativa Geográfica de Delitos por Departamento',
                template='plotly_white',
                height=300 * len(datos_por_depto),
                font=dict(family="Arial", size=12),
                title_font=dict(size=22),
                showlegend=False,
                plot_bgcolor='white'
            )
            
            # Actualizar ejes
            for i in range(1, len(datos_por_depto) + 1):
                fig.update_xaxes(title="Cantidad de Casos", row=i, col=1, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title="Departamento", row=i, col=1, autorange="reversed")
            
            fig.write_html('visualizaciones/comparativa_zonas_delitos.html')
            
            # Crear mapa combinado si hay datos de coordenadas para al menos un tipo de delito
            mapa_combinado = folium.Map(
                location=[4.570868, -74.297333],  # Coordenadas aproximadas de Colombia
                zoom_start=6,
                tiles='CartoDB positron'
            )
            
            colores_delitos = {
                'Homicidios': 'red',
                'Hurto a Personas': 'blue',
                'Hurto a Comercio': 'green',
                'Hurto de Automotores': 'orange'
            }
            
            for tipo, archivo in tipos_delitos.items():
                df = load_dataset(archivo)
                
                if df is not None:
                    # Buscar columnas de coordenadas
                    col_lat = next((col for col in df.columns if 'LAT' in col.upper()), None)
                    col_lon = next((col for col in df.columns if 'LON' in col.upper()), None)
                    
                    if col_lat and col_lon:
                        # Convertir a numérico y filtrar valores nulos y cero
                        df[col_lat] = pd.to_numeric(df[col_lat], errors='coerce')
                        df[col_lon] = pd.to_numeric(df[col_lon], errors='coerce')
                        
                        # Filtrar valores en Colombia (aproximadamente)
                        df_filtrado = df[
                            (df[col_lat].between(-4.2, 13.0)) & 
                            (df[col_lon].between(-82.0, -66.0))
                        ]
                        
                        if not df_filtrado.empty:
                            # Tomar muestra aleatoria para rendimiento
                            if len(df_filtrado) > 500:
                                df_filtrado = df_filtrado.sample(500, random_state=42)
                            
                            # Crear capa para este tipo de delito
                            feature_group = folium.FeatureGroup(name=tipo)
                            
                            # Añadir puntos
                            for idx, row in df_filtrado.iterrows():
                                folium.CircleMarker(
                                    [row[col_lat], row[col_lon]],
                                    radius=4,
                                    color=colores_delitos.get(tipo, 'gray'),
                                    fill=True,
                                    fill_color=colores_delitos.get(tipo, 'gray'),
                                    fill_opacity=0.7,
                                    popup=f"{tipo}<br>Lat: {row[col_lat]}<br>Lon: {row[col_lon]}"
                                ).add_to(feature_group)
                            
                            feature_group.add_to(mapa_combinado)
            
            # Añadir control de capas
            folium.LayerControl().add_to(mapa_combinado)
            
            # Guardar mapa
            mapa_combinado.save('visualizaciones/mapa_conjunto_delitos.html')
            
        return True
    except Exception as e:
        print(f"  Error al analizar zonas de delitos: {e}")
    
    return False

# ==========================================
# Cargar archivos y ejecutar análisis
# ==========================================

# Lista de archivos para análisis geográfico
archivos_analizar = [
    "Homicidios.csv",
    "Hurto_Personas.csv",
    "Hurto_Comercio.csv",
    "Hurto_Automotores.csv"
]

# Realizar análisis geográfico
for archivo in archivos_analizar:
    nombre = archivo.replace(".csv", "")
    df = load_dataset(archivo)
    if df is not None:
        # Buscar columnas de coordenadas
        cols_lat = [col for col in df.columns if 'LAT' in col.upper()]
        cols_lon = [col for col in df.columns if 'LON' in col.upper()]
        
        # Buscar columnas de departamento y municipio
        cols_depto = [col for col in df.columns if 'DEPART' in col.upper() or 'DEPTO' in col.upper()]
        cols_muni = [col for col in df.columns if 'MUNI' in col.upper() or 'CIUDAD' in col.upper()]
        
        if cols_lat and cols_lon and cols_depto:
            analizar_distribucion_geografica(df, nombre, cols_lat[0], cols_lon[0], cols_depto[0], 
                                            cols_muni[0] if cols_muni else None)

# Analizar frentes de seguridad
analizar_frentes_seguridad()

# Analizar zonas de delitos
analizar_zonas_delitos()

print("\nAnálisis geográfico completado. Visualizaciones guardadas en la carpeta 'visualizaciones'.") 