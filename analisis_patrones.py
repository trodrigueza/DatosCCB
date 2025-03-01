import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from datetime import datetime
import calendar
import warnings
warnings.filterwarnings('ignore')

# Set plot styles for better aesthetics
plt.style.use('ggplot')
sns.set_style('whitegrid')
sns.set_context('talk')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.family'] = 'sans-serif'

# Create output directory for visualizations if it doesn't exist
os.makedirs('visualizaciones', exist_ok=True)

print("Iniciando análisis de patrones de criminalidad...")

# Function to load a dataset
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

# ==========================================
# Análisis de patrones temporales detallados
# ==========================================

def analizar_patrones_hora_dia(df, nombre, col_fecha=None, col_hora=None):
    """Analiza patrones por hora del día y día de la semana"""
    try:
        print(f"\nAnalizando patrones horarios en {nombre}...")
        fecha_procesada = False
        
        # Caso 1: Columnas separadas de Año, Mes, Día
        if 'Año' in df.columns or 'AÑO' in df.columns:
            columna_anio = next((col for col in df.columns if col.upper() in ['AÑO', 'ANO', 'YEAR']), None)
            columna_mes = next((col for col in df.columns if col.upper() in ['MES', 'MONTH']), None)
            columna_dia = next((col for col in df.columns if col.upper() in ['DÍA', 'DIA', 'DAY']), None)
            
            if columna_anio and columna_mes:
                print(f"  Usando columnas separadas: {columna_anio}, {columna_mes}{', ' + columna_dia if columna_dia else ''}")
                
                # Convertir a valores numéricos
                df['año_num'] = pd.to_numeric(df[columna_anio], errors='coerce')
                df['mes_num'] = pd.Series(pd.to_datetime(df[columna_mes], format='%B', errors='coerce').dt.month)
                
                if columna_dia:
                    # Intentar diferentes formatos para el día
                    try:
                        # Primero intentar como día de la semana (lun, mar, etc)
                        dias_semana_map = {'lun': 0, 'mar': 1, 'mié': 2, 'jue': 3, 'vie': 4, 'sáb': 5, 'dom': 6}
                        df['dia_semana'] = df[columna_dia].str.lower().str[:3].map(dias_semana_map)
                    except:
                        # Si falla, intentar como número de día del mes
                        df['dia_num'] = pd.to_numeric(df[columna_dia], errors='coerce')
                
                # Análisis por mes
                if not df['mes_num'].isna().all():
                    meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                             'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
                    mes_counts = df['mes_num'].value_counts().sort_index().reset_index()
                    mes_counts.columns = ['mes', 'cantidad']
                    mes_counts['nombre_mes'] = mes_counts['mes'].apply(
                        lambda x: meses[int(x)-1] if pd.notna(x) and isinstance(x, (int, float)) and 1 <= int(x) <= 12 else 'Desconocido'
                    )
                    
                    fig = px.bar(mes_counts, x='nombre_mes', y='cantidad',
                                 title=f'Incidencia por Mes - {nombre}',
                                 labels={'cantidad': 'Cantidad de Casos', 'nombre_mes': 'Mes'},
                                 color='cantidad',
                                 color_continuous_scale='Viridis')
                    
                    fig.update_layout(
                        template='plotly_white',
                        plot_bgcolor='white',
                        font=dict(family="Arial", size=12),
                        title=dict(font=dict(size=20)),
                        xaxis=dict(showgrid=True, gridcolor='lightgray', categoryorder='array', categoryarray=meses),
                        yaxis=dict(showgrid=True, gridcolor='lightgray')
                    )
                    
                    fig.write_html(f'visualizaciones/{nombre}_patrones_mes.html')
                    fecha_procesada = True
                
                # Análisis por día de la semana si se pudo procesar
                if 'dia_semana' in df.columns and not df['dia_semana'].isna().all():
                    dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                    dia_counts = df['dia_semana'].value_counts().sort_index().reset_index()
                    dia_counts.columns = ['dia_semana', 'cantidad']
                    dia_counts['nombre_dia'] = dia_counts['dia_semana'].apply(lambda x: dias_semana[x] if 0 <= x < 7 else 'Desconocido')
                    
                    fig = px.bar(dia_counts, x='nombre_dia', y='cantidad',
                                 title=f'Incidencia por Día de la Semana - {nombre}',
                                 labels={'cantidad': 'Cantidad de Casos', 'nombre_dia': 'Día de la Semana'},
                                 color='cantidad',
                                 color_continuous_scale='Viridis')
                    
                    fig.update_layout(
                        template='plotly_white',
                        plot_bgcolor='white',
                        font=dict(family="Arial", size=12),
                        title=dict(font=dict(size=20)),
                        xaxis=dict(showgrid=True, gridcolor='lightgray', categoryorder='array', categoryarray=dias_semana),
                        yaxis=dict(showgrid=True, gridcolor='lightgray')
                    )
                    
                    fig.write_html(f'visualizaciones/{nombre}_patrones_dia_semana.html')
                    fecha_procesada = True
        
        # Caso 2: Una sola columna de fecha
        if not fecha_procesada and col_fecha and col_fecha in df.columns:
            # Convertir fecha
            print(f"  Usando columna de fecha: {col_fecha}")
            df[col_fecha] = pd.to_datetime(df[col_fecha], errors='coerce')
            
            # Extraer información temporal
            df['dia_semana'] = df[col_fecha].dt.dayofweek
            df['mes'] = df[col_fecha].dt.month
            
            # Análisis por día de la semana
            dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
            dia_counts = df['dia_semana'].value_counts().sort_index().reset_index()
            dia_counts.columns = ['dia_semana', 'cantidad']
            dia_counts['nombre_dia'] = dia_counts['dia_semana'].apply(
                lambda x: dias_semana[int(x)] if pd.notna(x) and isinstance(x, (int, float)) and 0 <= int(x) < 7 else 'Desconocido'
            )
            
            fig = px.bar(dia_counts, x='nombre_dia', y='cantidad',
                         title=f'Incidencia por Día de la Semana - {nombre}',
                         labels={'cantidad': 'Cantidad de Casos', 'nombre_dia': 'Día de la Semana'},
                         color='cantidad',
                         color_continuous_scale='Viridis')
            
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                font=dict(family="Arial", size=12),
                title=dict(font=dict(size=20)),
                xaxis=dict(showgrid=True, gridcolor='lightgray', categoryorder='array', categoryarray=dias_semana),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )
            
            fig.write_html(f'visualizaciones/{nombre}_patrones_dia_semana.html')
            
            # Análisis por mes
            meses = [calendar.month_name[i] for i in range(1, 13)]
            mes_counts = df['mes'].value_counts().sort_index().reset_index()
            mes_counts.columns = ['mes', 'cantidad']
            mes_counts['nombre_mes'] = mes_counts['mes'].apply(
                lambda x: meses[int(x)-1] if pd.notna(x) and isinstance(x, (int, float)) and 1 <= int(x) <= 12 else 'Desconocido'
            )
            
            fig = px.bar(mes_counts, x='nombre_mes', y='cantidad',
                         title=f'Incidencia por Mes - {nombre}',
                         labels={'cantidad': 'Cantidad de Casos', 'nombre_mes': 'Mes'},
                         color='cantidad',
                         color_continuous_scale='Viridis')
            
            fig.update_layout(
                template='plotly_white',
                plot_bgcolor='white',
                font=dict(family="Arial", size=12),
                title=dict(font=dict(size=20)),
                xaxis=dict(showgrid=True, gridcolor='lightgray', categoryorder='array', categoryarray=meses),
                yaxis=dict(showgrid=True, gridcolor='lightgray')
            )
            
            fig.write_html(f'visualizaciones/{nombre}_patrones_mes.html')
            
            fecha_procesada = True
            
            # Análisis por hora si existe la columna
            if col_hora in df.columns:
                # Intentar convertir a formato de hora
                try:
                    # Procesar diferentes formatos posibles de hora
                    if df[col_hora].dtype == 'object':
                        # Intentar extraer la hora (primera parte antes de :)
                        df['hora_num'] = df[col_hora].str.extract(r'(\d+)').astype(float)
                    else:
                        df['hora_num'] = df[col_hora].astype(float)
                    
                    # Filtrar horas válidas (0-23)
                    df_horas = df[df['hora_num'].between(0, 23)]
                    
                    if not df_horas.empty:
                        hora_counts = df_horas['hora_num'].value_counts().sort_index().reset_index()
                        hora_counts.columns = ['hora', 'cantidad']
                        
                        # Crear visualización de patrones por hora
                        fig = px.line(hora_counts, x='hora', y='cantidad',
                                     title=f'Incidencia por Hora del Día - {nombre}',
                                     labels={'cantidad': 'Cantidad de Casos', 'hora': 'Hora del Día'},
                                     markers=True)
                        
                        fig.update_layout(
                            template='plotly_white',
                            plot_bgcolor='white',
                            font=dict(family="Arial", size=12),
                            title=dict(font=dict(size=20)),
                            xaxis=dict(
                                showgrid=True, 
                                gridcolor='lightgray',
                                tickmode='array',
                                tickvals=list(range(0, 24)),
                                ticktext=[f"{i}:00" for i in range(24)]
                            ),
                            yaxis=dict(showgrid=True, gridcolor='lightgray')
                        )
                        
                        fig.write_html(f'visualizaciones/{nombre}_patrones_hora.html')
                        
                        # Heatmap de día de la semana vs hora
                        pivot = pd.crosstab(df_horas['dia_semana'], df_horas['hora_num'])
                        
                        fig = px.imshow(pivot, 
                                       labels=dict(x="Hora del día", y="Día de la semana", color="Cantidad"),
                                       x=[f"{i}:00" for i in range(24)],
                                       y=dias_semana,
                                       title=f'Patrón Semanal por Hora - {nombre}',
                                       color_continuous_scale='Viridis')
                        
                        fig.update_layout(
                            template='plotly_white',
                            font=dict(family="Arial", size=12),
                            title=dict(font=dict(size=20))
                        )
                        
                        fig.write_html(f'visualizaciones/{nombre}_heatmap_dia_hora.html')
                except Exception as e:
                    print(f"  Error al procesar datos de hora en {nombre}: {e}")
        
        if fecha_procesada:
            return True
        else:
            print(f"  No se encontraron columnas de fecha/tiempo válidas en {nombre}")
            return False
    except Exception as e:
        print(f"  Error al analizar patrones temporales en {nombre}: {e}")
        return False

# ==========================================
# Análisis comparativo entre tipos de delitos
# ==========================================

def comparativa_delitos():
    """Genera una comparativa entre diferentes tipos de delitos considerando solo municipios de Cundinamarca"""
    print("\nGenerando comparativa entre tipos de delitos por municipio en Cundinamarca...")
    
    try:
        # Comparar homicidios, hurto a personas, hurto a comercio y hurto de automotores
        datasets_comparar = {
            'Homicidios': 'homicidios',
            'Hurto_Personas': 'hurto_personas',
            'Hurto_Comercio': 'hurto_comercio',
            'Hurto_Automotores': 'hurto_automotores'
        }
        
        # Para almacenar datos por municipio
        datos_municipios = {}
        datos_categorias = {}
        
        for nombre_archivo, etiqueta in datasets_comparar.items():
            try:
                df = load_dataset(f"{nombre_archivo}.csv")
                if df is not None:
                    # Filtrar solo para departamento de Cundinamarca
                    columnas_depto = [col for col in df.columns if 'DEPART' in col.upper() or 'DEPTO' in col.upper()]
                    if columnas_depto:
                        col_depto = columnas_depto[0]
                        df_cundinamarca = df[df[col_depto].str.upper().str.contains('CUNDI', na=False)]
                        
                        if df_cundinamarca.empty:
                            print(f"  No se encontraron datos de Cundinamarca en {nombre_archivo}")
                            continue
                        
                        print(f"  Analizando {nombre_archivo} - Datos de Cundinamarca: {df_cundinamarca.shape[0]} registros")
                        
                        # Análisis por municipio
                        columnas_muni = [col for col in df_cundinamarca.columns if 'MUNI' in col.upper() or 'CIUDAD' in col.upper()]
                        if columnas_muni:
                            col_muni = columnas_muni[0]
                            
                            # Contar por municipio
                            muni_counts = df_cundinamarca[col_muni].value_counts().reset_index()
                            muni_counts.columns = [col_muni, 'cantidad']
                            muni_counts = muni_counts.sort_values('cantidad', ascending=False).head(15)  # Top 15 municipios
                            
                            datos_municipios[etiqueta] = {
                                'columna': col_muni,
                                'data': muni_counts
                            }
                        
                        # Análisis por categoría (modalidad, tipo de delito, etc.)
                        for posible_cat in ['MODALIDAD', 'TIPO', 'ARMAS MEDIOS', 'Armas / Medios', 'GENERO', 'DELITO']:
                            if posible_cat in df_cundinamarca.columns:
                                cat_counts = df_cundinamarca[posible_cat].value_counts().reset_index()
                                cat_counts.columns = [posible_cat, 'cantidad']
                                cat_counts = cat_counts.sort_values('cantidad', ascending=False).head(10)  # Top 10 categorías
                                
                                # Almacenar para visualización
                                clave_categoria = f"{etiqueta}_{posible_cat}"
                                datos_categorias[clave_categoria] = {
                                    'nombre': nombre_archivo.replace('_', ' '),
                                    'columna': posible_cat,
                                    'data': cat_counts
                                }
                                break  # Solo usar la primera categoría encontrada
            except Exception as e:
                print(f"  Error al procesar {nombre_archivo}: {e}")
        
        # Crear visualizaciones de municipios
        if datos_municipios:
            fig = make_subplots(rows=len(datos_municipios), cols=1, 
                               subplot_titles=[f"Top 15 Municipios - {tipo.replace('_', ' ').title()}" for tipo in datos_municipios.keys()],
                               vertical_spacing=0.08)
            
            # Añadir cada tipo de delito como subplot
            for i, (tipo, datos) in enumerate(datos_municipios.items(), 1):
                col_muni = datos['columna']
                muni_data = datos['data']
                
                fig.add_trace(
                    go.Bar(
                        y=muni_data[col_muni],
                        x=muni_data['cantidad'],
                        name=tipo.replace('_', ' ').title(),
                        orientation='h',
                        marker=dict(color=px.colors.qualitative.Plotly[i-1])
                    ),
                    row=i, col=1
                )
            
            fig.update_layout(
                title='Comparativa de Delitos por Municipio en Cundinamarca',
                template='plotly_white',
                height=300 * len(datos_municipios),
                width=1000,
                font=dict(family="Arial", size=12),
                title_font=dict(size=22),
                showlegend=False,
                plot_bgcolor='white'
            )
            
            # Actualizar ejes
            for i in range(1, len(datos_municipios) + 1):
                fig.update_xaxes(title="Cantidad de Casos", row=i, col=1, showgrid=True, gridcolor='lightgray')
                fig.update_yaxes(title="Municipio", row=i, col=1)
            
            fig.write_html('visualizaciones/comparativa_delitos_municipios.html')
        
        # Crear visualizaciones de categorías
        for clave, datos in datos_categorias.items():
            nombre = datos['nombre']
            col_cat = datos['columna']
            cat_data = datos['data']
            
            fig = px.bar(cat_data, y=col_cat, x='cantidad',
                        title=f'Principales {col_cat} en {nombre}',
                        labels={'cantidad': 'Cantidad de Casos', col_cat: col_cat},
                        color='cantidad',
                        color_continuous_scale='Viridis',
                        orientation='h')
            
            fig.update_layout(
                template='plotly_white',
                height=600,
                width=1000,
                font=dict(family="Arial", size=12),
                title_font=dict(size=20),
                plot_bgcolor='white',
                xaxis=dict(showgrid=True, gridcolor='lightgray'),
                yaxis=dict(categoryorder='total ascending')
            )
            
            # Guardar con nombre normalizado
            nombre_archivo = f'visualizaciones/{clave.replace(" ", "_").lower()}_categorias.html'
            fig.write_html(nombre_archivo)
            print(f"  Visualización de categorías guardada como {nombre_archivo}")
        
        # Análisis combinado de municipios con mayor incidencia
        try:
            print("  Generando análisis combinado de municipios...")
            # Extraer todos los municipios y sus conteos por tipo de delito
            all_munis = {}
            
            for tipo, datos in datos_municipios.items():
                col_muni = datos['columna']
                muni_data = datos['data']
                
                for _, row in muni_data.iterrows():
                    muni = row[col_muni]
                    if muni not in all_munis:
                        all_munis[muni] = {}
                    all_munis[muni][tipo] = row['cantidad']
            
            # Convertir a DataFrame
            rows = []
            for muni, counts in all_munis.items():
                row = {'Municipio': muni}
                row.update(counts)
                rows.append(row)
            
            df_combined = pd.DataFrame(rows)
            df_combined.fillna(0, inplace=True)
            
            # Calcular total y ordenar
            for tipo in datos_municipios.keys():
                if tipo not in df_combined.columns:
                    df_combined[tipo] = 0
            
            df_combined['Total'] = sum(df_combined[tipo] for tipo in datos_municipios.keys())
            df_combined = df_combined.sort_values('Total', ascending=False).head(10)  # Top 10 municipios
            
            # Crear gráfico de barras apiladas
            fig = go.Figure()
            
            for tipo in datos_municipios.keys():
                fig.add_trace(go.Bar(
                    name=tipo.replace('_', ' ').title(),
                    x=df_combined['Municipio'],
                    y=df_combined[tipo],
                    marker=dict(
                        color=px.colors.qualitative.Plotly[list(datos_municipios.keys()).index(tipo)]
                    )
                ))
            
            fig.update_layout(
                title='Top 10 Municipios con Mayor Incidencia de Delitos Combinados',
                template='plotly_white',
                barmode='stack',
                height=600,
                width=1000,
                font=dict(family="Arial", size=12),
                title_font=dict(size=22),
                plot_bgcolor='white',
                legend_title_text='Tipo de Delito',
                xaxis=dict(title="Municipio", showgrid=True, gridcolor='lightgray', tickangle=45),
                yaxis=dict(title="Cantidad de Casos", showgrid=True, gridcolor='lightgray')
            )
            
            fig.write_html('visualizaciones/top_municipios_delitos_combinados.html')
            
        except Exception as e:
            print(f"  Error al generar análisis combinado: {e}")
        
        return True
    except Exception as e:
        print(f"  Error en comparativa de delitos: {e}")
        return False

# ==========================================
# Análisis de presupuesto vs incidencia
# ==========================================

def analizar_presupuesto_vs_delitos():
    """Analiza la relación entre presupuesto y niveles de criminalidad"""
    print("\nAnalizando relación entre presupuesto y delitos...")
    
    try:
        # Cargar datos de presupuesto
        df_presupuesto = load_dataset("Presupuesto_de_Gastos.csv")
        
        if df_presupuesto is not None:
            # Verificar si existe columna de año y valor
            columnas_año = [col for col in df_presupuesto.columns if 'AÑO' in col.upper() or 'VIGENCIA' in col.upper()]
            columnas_valor = [col for col in df_presupuesto.columns if 'VALOR' in col.upper() or 'MONTO' in col.upper() or 'PRESUPUESTO' in col.upper()]
            
            if columnas_año and columnas_valor:
                # Usar las primeras columnas encontradas
                col_año = columnas_año[0]
                col_valor = columnas_valor[0]
                
                # Agrupar por año y sumar valores
                df_presupuesto_anual = df_presupuesto.groupby(col_año)[col_valor].sum().reset_index()
                df_presupuesto_anual.columns = ['año', 'presupuesto']
                
                # Cargar datos de delitos (usar homicidios como ejemplo)
                df_homicidios = load_dataset("Homicidios.csv")
                
                if df_homicidios is not None and 'FECHA' in df_homicidios.columns:
                    # Procesar fecha
                    df_homicidios['FECHA'] = pd.to_datetime(df_homicidios['FECHA'], errors='coerce')
                    df_homicidios['año'] = df_homicidios['FECHA'].dt.year
                    
                    # Contar homicidios por año
                    homicidios_anual = df_homicidios.groupby('año').size().reset_index(name='homicidios')
                    
                    # Unir con datos de presupuesto
                    df_combinado = pd.merge(df_presupuesto_anual, homicidios_anual, on='año', how='inner')
                    
                    if not df_combinado.empty:
                        # Crear visualización
                        fig = make_subplots(specs=[[{"secondary_y": True}]])
                        
                        fig.add_trace(
                            go.Bar(x=df_combinado['año'], y=df_combinado['presupuesto'], 
                                  name="Presupuesto", marker_color='green'),
                            secondary_y=False
                        )
                        
                        fig.add_trace(
                            go.Scatter(x=df_combinado['año'], y=df_combinado['homicidios'], 
                                      name="Homicidios", line=dict(color='red', width=3),
                                      mode='lines+markers'),
                            secondary_y=True
                        )
                        
                        fig.update_layout(
                            title='Relación entre Presupuesto y Homicidios por Año',
                            template='plotly_white',
                            font=dict(family="Arial", size=12),
                            title_font=dict(size=20),
                            plot_bgcolor='white',
                            legend_title_text='',
                            barmode='group',
                            xaxis=dict(title="Año", showgrid=True, gridcolor='lightgray')
                        )
                        
                        fig.update_yaxes(title_text="Presupuesto", secondary_y=False)
                        fig.update_yaxes(title_text="Número de Homicidios", secondary_y=True)
                        
                        fig.write_html('visualizaciones/presupuesto_vs_homicidios.html')
                        
                        # Calcular correlación
                        corr = df_combinado['presupuesto'].corr(df_combinado['homicidios'])
                        print(f"  Correlación entre presupuesto y homicidios: {corr:.2f}")
                        
                        return True
    except Exception as e:
        print(f"  Error al analizar presupuesto vs delitos: {e}")
    
    return False

# ==========================================
# Cargar datasets relevantes para análisis de patrones
# ==========================================

# Lista de archivos para analizar patrones
archivos_analizar = [
    "Hurto_Personas.csv",
    "Homicidios.csv",
    "Hurto_Comercio.csv",
    "Hurto_Automotores.csv",
    "Violencia_Intrafamiliar.csv"
]

# Realizar análisis de patrones temporales
for archivo in archivos_analizar:
    nombre = archivo.replace(".csv", "")
    df = load_dataset(archivo)
    if df is not None:
        patron_encontrado = False
        
        # Caso 1: Verificar si hay columnas separadas Año, Mes, Día (con manejo de codificación)
        # Buscar todas las columnas que podrían ser año, mes o día
        posibles_cols_anio = [col for col in df.columns if 'AÃ±o' in col or 'Año' in col or 'ANO' in col.upper() or 'YEAR' in col.upper()]
        posibles_cols_mes = [col for col in df.columns if 'Mes' in col or 'MES' in col.upper() or 'MONTH' in col.upper()]
        posibles_cols_dia = [col for col in df.columns if 'DÃ­a' in col or 'Día' in col or 'DIA' in col.upper() or 'DAY' in col.upper()]
        
        print(f"\nAnalizando patrones temporales en {nombre}...")
        print(f"  Columnas de año encontradas: {posibles_cols_anio}")
        print(f"  Columnas de mes encontradas: {posibles_cols_mes}")
        print(f"  Columnas de día encontradas: {posibles_cols_dia}")
        
        if posibles_cols_anio:
            print(f"  Usando columnas separadas para análisis temporal")
            # Buscar columna de hora
            cols_hora = [col for col in df.columns if 'HORA' in col.upper() or 'RANGO_HORARIO' in col.upper()]
            
            # Asignar manualmente las columnas para el análisis
            df_analisis = df.copy()
            
            # Convertir año a numérico
            anio_col = posibles_cols_anio[0]
            df_analisis['año_num'] = pd.to_numeric(df_analisis[anio_col], errors='coerce')
            
            # Procesar mes (pueden ser nombres o números)
            if posibles_cols_mes:
                mes_col = posibles_cols_mes[0]
                try:
                    # Primero intentar convertir directamente a número
                    df_analisis['mes_num'] = pd.to_numeric(df_analisis[mes_col], errors='coerce')
                    
                    # Si no funcionó (mayoría son NaN), intentar convertir desde nombres de mes
                    if df_analisis['mes_num'].isna().mean() > 0.5:
                        # Mapa de nombres de mes a números
                        meses_map = {
                            'ENERO': 1, 'FEBRERO': 2, 'MARZO': 3, 'ABRIL': 4, 'MAYO': 5, 'JUNIO': 6,
                            'JULIO': 7, 'AGOSTO': 8, 'SEPTIEMBRE': 9, 'OCTUBRE': 10, 'NOVIEMBRE': 11, 'DICIEMBRE': 12,
                            'Enero': 1, 'Febrero': 2, 'Marzo': 3, 'Abril': 4, 'Mayo': 5, 'Junio': 6,
                            'Julio': 7, 'Agosto': 8, 'Septiembre': 9, 'Octubre': 10, 'Noviembre': 11, 'Diciembre': 12
                        }
                        df_analisis['mes_num'] = df_analisis[mes_col].map(meses_map)
                except:
                    print(f"  Error al procesar columna de mes: {mes_col}")
            
            # Procesar día si existe (podría ser día de semana o número)
            if posibles_cols_dia:
                dia_col = posibles_cols_dia[0]
                try:
                    # Primero verificar si son días de la semana abreviados
                    dias_semana_map = {
                        'lun.': 0, 'mar.': 1, 'miÃ©.': 2, 'mié.': 2, 'jue.': 3, 'vie.': 4, 'sÃ¡b.': 5, 'sáb.': 5, 'dom.': 6,
                        'lun': 0, 'mar': 1, 'mié': 2, 'miÃ©': 2, 'jue': 3, 'vie': 4, 'sáb': 5, 'sÃ¡b': 5, 'dom': 6,
                        'Lun': 0, 'Mar': 1, 'Mié': 2, 'MiÃ©': 2, 'Jue': 3, 'Vie': 4, 'Sáb': 5, 'SÃ¡b': 5, 'Dom': 6,
                        'Lunes': 0, 'Martes': 1, 'Miércoles': 2, 'MiÃ©rcoles': 2, 'Jueves': 3, 'Viernes': 4, 
                        'Sábado': 5, 'SÃ¡bado': 5, 'Domingo': 6
                    }
                    
                    # Intentar mapear nombres de día a números de día de semana
                    if df_analisis[dia_col].dtype == 'object':
                        df_analisis['dia_semana'] = df_analisis[dia_col].map(dias_semana_map)
                    else:
                        # Si es numérico, asumir que es día del mes, no día de la semana
                        pass
                except:
                    print(f"  Error al procesar columna de día: {dia_col}")
            
            # Ahora realizar los análisis de patrones
            
            # Análisis por mes
            if 'mes_num' in df_analisis.columns and not df_analisis['mes_num'].isna().all():
                meses = ['Enero', 'Febrero', 'Marzo', 'Abril', 'Mayo', 'Junio', 
                         'Julio', 'Agosto', 'Septiembre', 'Octubre', 'Noviembre', 'Diciembre']
                mes_counts = df_analisis['mes_num'].value_counts().sort_index().reset_index()
                mes_counts.columns = ['mes', 'cantidad']
                mes_counts['nombre_mes'] = mes_counts['mes'].apply(
                    lambda x: meses[int(x)-1] if pd.notna(x) and isinstance(x, (int, float)) and 1 <= int(x) <= 12 else 'Desconocido'
                )
                
                fig = px.bar(mes_counts, x='nombre_mes', y='cantidad',
                             title=f'Incidencia por Mes - {nombre}',
                             labels={'cantidad': 'Cantidad de Casos', 'nombre_mes': 'Mes'},
                             color='cantidad',
                             color_continuous_scale='Viridis')
                
                fig.update_layout(
                    template='plotly_white',
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=12),
                    title=dict(font=dict(size=20)),
                    xaxis=dict(showgrid=True, gridcolor='lightgray', categoryorder='array', categoryarray=meses),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                
                fig.write_html(f'visualizaciones/{nombre}_patrones_mes.html')
                patron_encontrado = True
            
            # Análisis por día de la semana si se pudo procesar
            if 'dia_semana' in df_analisis.columns and not df_analisis['dia_semana'].isna().all():
                dias_semana = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']
                dia_counts = df_analisis['dia_semana'].value_counts().sort_index().reset_index()
                dia_counts.columns = ['dia_semana', 'cantidad']
                dia_counts['nombre_dia'] = dia_counts['dia_semana'].apply(
                    lambda x: dias_semana[int(x)] if pd.notna(x) and isinstance(x, (int, float)) and 0 <= int(x) < 7 else 'Desconocido'
                )
                
                fig = px.bar(dia_counts, x='nombre_dia', y='cantidad',
                             title=f'Incidencia por Día de la Semana - {nombre}',
                             labels={'cantidad': 'Cantidad de Casos', 'nombre_dia': 'Día de la Semana'},
                             color='cantidad',
                             color_continuous_scale='Viridis')
                
                fig.update_layout(
                    template='plotly_white',
                    plot_bgcolor='white',
                    font=dict(family="Arial", size=12),
                    title=dict(font=dict(size=20)),
                    xaxis=dict(showgrid=True, gridcolor='lightgray', categoryorder='array', categoryarray=dias_semana),
                    yaxis=dict(showgrid=True, gridcolor='lightgray')
                )
                
                fig.write_html(f'visualizaciones/{nombre}_patrones_dia_semana.html')
                patron_encontrado = True
            
            # Análisis por hora si hay columna disponible
            if cols_hora:
                hora_col = cols_hora[0]
                try:
                    # Procesar diferentes formatos posibles de hora
                    if df[hora_col].dtype == 'object':
                        # Intentar extraer la hora (primera parte antes de :)
                        df_analisis['hora_num'] = df[hora_col].str.extract(r'(\d+)').astype(float)
                    else:
                        df_analisis['hora_num'] = df[hora_col].astype(float)
                    
                    # Filtrar horas válidas (0-23)
                    df_horas = df_analisis[df_analisis['hora_num'].between(0, 23)]
                    
                    if not df_horas.empty:
                        hora_counts = df_horas['hora_num'].value_counts().sort_index().reset_index()
                        hora_counts.columns = ['hora', 'cantidad']
                        
                        # Crear visualización de patrones por hora
                        fig = px.line(hora_counts, x='hora', y='cantidad',
                                     title=f'Incidencia por Hora del Día - {nombre}',
                                     labels={'cantidad': 'Cantidad de Casos', 'hora': 'Hora del Día'},
                                     markers=True)
                        
                        fig.update_layout(
                            template='plotly_white',
                            plot_bgcolor='white',
                            font=dict(family="Arial", size=12),
                            title=dict(font=dict(size=20)),
                            xaxis=dict(
                                showgrid=True, 
                                gridcolor='lightgray',
                                tickmode='array',
                                tickvals=list(range(0, 24)),
                                ticktext=[f"{i}:00" for i in range(24)]
                            ),
                            yaxis=dict(showgrid=True, gridcolor='lightgray')
                        )
                        
                        fig.write_html(f'visualizaciones/{nombre}_patrones_hora.html')
                        patron_encontrado = True
                except Exception as e:
                    print(f"  Error al procesar datos de hora en {nombre}: {e}")
        
        # Caso 2: Intentar diferentes combinaciones de columnas de fecha y hora
        if not patron_encontrado:
            cols_fecha = ['FECHA', 'FECHA_HECHO', 'FECHA HECHO', 'FECHA_COMISION', 'FECHA COMISION']
            cols_hora = ['HORA', 'HORA_HECHO', 'HORA HECHO', 'HORA_COMISION', 'HORA COMISION', 'RANGO_HORARIO']
            
            for col_fecha in cols_fecha:
                if col_fecha in df.columns:
                    for col_hora in cols_hora:
                        if col_hora in df.columns:
                            patron_encontrado = analizar_patrones_hora_dia(df, nombre, col_fecha, col_hora)
                            break
                    
                    if not patron_encontrado:
                        # Si no encontró columna de hora, analizar solo con fecha
                        patron_encontrado = analizar_patrones_hora_dia(df, nombre, col_fecha)
                    
                    if patron_encontrado:
                        break

# Realizar análisis comparativo entre delitos
comparativa_delitos()

# Analizar relación entre presupuesto y delitos
analizar_presupuesto_vs_delitos()

print("\nAnálisis de patrones completado. Visualizaciones guardadas en la carpeta 'visualizaciones'.") 