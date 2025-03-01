import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import time
import subprocess
import datetime
import warnings
warnings.filterwarnings('ignore')

print("=" * 80)
print("ANÁLISIS INTEGRAL DE DATOS DE SEGURIDAD Y CRIMINALIDAD")
print("=" * 80)
print(f"Fecha de ejecución: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)

# Crear directorio de visualizaciones si no existe
os.makedirs('visualizaciones', exist_ok=True)

# Crear directorio para el informe
os.makedirs('informe', exist_ok=True)

# Función para ejecutar un script y medir tiempo
def ejecutar_script(nombre_script):
    print(f"\nEjecutando {nombre_script}...")
    tiempo_inicio = time.time()
    
    try:
        subprocess.run(['python', nombre_script], check=True)
        tiempo_fin = time.time()
        tiempo_total = tiempo_fin - tiempo_inicio
        print(f"Completado {nombre_script} en {tiempo_total:.2f} segundos")
        return True
    except Exception as e:
        print(f"Error al ejecutar {nombre_script}: {e}")
        return False

# Ejecutar los scripts de análisis en secuencia
scripts = [
    'analisis_seguridad.py',
    'analisis_patrones.py',
    'analisis_geografico.py',
    'fix_geographical_maps.py'  # Agregado script para generar los mapas
]

resultados_scripts = {}
for script in scripts:
    resultado = ejecutar_script(script)
    resultados_scripts[script] = resultado

# ==========================================
# Generar informe HTML con los resultados
# ==========================================

def generar_informe_html():
    print("\nGenerando informe HTML con los resultados del análisis...")
    
    # Obtener lista de visualizaciones generadas
    visualizaciones = [f for f in os.listdir('visualizaciones') if f.endswith('.html')]
    
    # Modificado para mantener solo el mapa de frentes de Bogotá y filtrar otros mapas
    visualizaciones_filtradas = []
    for v in visualizaciones:
        # Mantener solo el mapa de frentes de seguridad de Bogotá y cualquier visualización que no sea un mapa
        if 'Frentes_Seguridad_Bogota' in v:
            visualizaciones_filtradas.append(v)
        # Incluir específicamente la comparativa de municipios
        elif 'comparativa_delitos_municipios' in v:
            visualizaciones_filtradas.append(v)
        # Excluir específicamente la comparativa de zonas/departamentos
        elif 'comparativa_zonas_delitos' in v:
            continue
        elif not any(term in v.lower() for term in ['mapa', 'depto', 'geogr', 'municipio']):
            visualizaciones_filtradas.append(v)
    
    # Reemplazar la lista original con la filtrada
    visualizaciones = visualizaciones_filtradas
    
    # Agrupar visualizaciones por categoría
    categorias = {
        'Tendencias Temporales': [v for v in visualizaciones if 'tendencia' in v.lower() or 'patron' in v.lower()],
        'Frentes de Seguridad': [v for v in visualizaciones if 'frente' in v.lower() or 'Frentes_Seguridad_Bogota' in v],
        'Comparativas': [v for v in visualizaciones if 'compar' in v.lower() or 'vs' in v.lower()],
        'Análisis por Delito': [v for v in visualizaciones if any(delito in v.lower() for delito in ['homicidio', 'hurto', 'violencia'])]
    }
    
    # Eliminar la categoría 'Distribución Geográfica' que contiene otros mapas
    if 'Distribución Geográfica' in categorias:
        del categorias['Distribución Geográfica']
    
    # Crear HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Análisis de Datos de Seguridad y Criminalidad</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                text-align: center;
            }}
            h2 {{
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            .header {{
                background-color: #3498db;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .section {{
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 30px;
            }}
            .viz-container {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                justify-content: center;
            }}
            .viz-item {{
                width: calc(50% - 20px);
                margin-bottom: 20px;
                background-color: #f1f2f6;
                border-radius: 5px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .viz-item h4 {{
                margin: 0;
                padding: 10px;
                background-color: #dfe4ea;
                font-size: 14px;
                text-align: center;
            }}
            iframe {{
                border: none;
                width: 100%;
                height: 400px;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #bdc3c7;
                color: #7f8c8d;
            }}
            .resumen {{
                padding: 15px;
                background-color: #e8f4f8;
                border-left: 4px solid #3498db;
                margin-bottom: 20px;
            }}
            @media (max-width: 768px) {{
                .viz-item {{
                    width: 100%;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Análisis de Datos de Seguridad y Criminalidad</h1>
            <p>Fecha de generación: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Introducción</h2>
            <p>Este informe presenta un análisis de diversos conjuntos de datos relacionados con la seguridad y la criminalidad. 
            El análisis incluye tendencias temporales, patrones geográficos, y correlaciones entre diferentes tipos de delitos y variables.</p>
        </div>
    """
    
    # Agregar secciones para cada categoría
    for categoria, archivos in categorias.items():
        if archivos:
            html_content += f"""
            <div class="section">
                <h2>{categoria}</h2>
                <div class="viz-container">
            """
            
            for archivo in archivos:
                nombre_visual = archivo.replace('.html', '').replace('_', ' ').title()
                html_content += f"""
                    <div class="viz-item">
                        <h4>{nombre_visual}</h4>
                        <iframe src="../visualizaciones/{archivo}"></iframe>
                    </div>
                """
            
            html_content += """
                </div>
            </div>
            """
    
    # Conclusiones
    html_content += """
        <div class="footer">
            <p>Análisis realizado con Python utilizando pandas, matplotlib, seaborn, plotly y folium.</p>
            <p>Análisis de Datos de Seguridad y Criminalidad</p>
        </div>
    </body>
    </html>
    """
    
    # Guardar archivo HTML
    with open('informe/reporte_analisis.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Informe HTML generado exitosamente en 'informe/reporte_analisis.html'")

# Generar informe HTML
generar_informe_html()

# ==========================================
# Generar informe HTML con visualizaciones a pantalla completa
# ==========================================

def generar_informe_pantalla_completa():
    print("\nGenerando informe HTML con visualizaciones a pantalla completa...")
    
    # Obtener lista de visualizaciones generadas (usar el mismo filtrado que el informe principal)
    visualizaciones = [f for f in os.listdir('visualizaciones') if f.endswith('.html')]
    
    # Modificado para mantener solo el mapa de frentes de Bogotá y filtrar otros mapas
    visualizaciones_filtradas = []
    for v in visualizaciones:
        # Mantener solo el mapa de frentes de seguridad de Bogotá y cualquier visualización que no sea un mapa
        if 'Frentes_Seguridad_Bogota' in v:
            visualizaciones_filtradas.append(v)
        # Incluir específicamente la comparativa de municipios
        elif 'comparativa_delitos_municipios' in v:
            visualizaciones_filtradas.append(v)
        # Excluir específicamente la comparativa de zonas/departamentos
        elif 'comparativa_zonas_delitos' in v:
            continue
        elif not any(term in v.lower() for term in ['mapa', 'depto', 'geogr', 'municipio']):
            visualizaciones_filtradas.append(v)
    
    # Reemplazar la lista original con la filtrada
    visualizaciones = visualizaciones_filtradas
    
    # Agrupar visualizaciones por categoría
    categorias = {
        'Tendencias Temporales': [v for v in visualizaciones if 'tendencia' in v.lower() or 'patron' in v.lower()],
        'Frentes de Seguridad': [v for v in visualizaciones if 'frente' in v.lower() or 'Frentes_Seguridad_Bogota' in v],
        'Comparativas': [v for v in visualizaciones if 'compar' in v.lower() or 'vs' in v.lower()],
        'Análisis por Delito': [v for v in visualizaciones if any(delito in v.lower() for delito in ['homicidio', 'hurto', 'violencia'])]
    }
    
    # Eliminar la categoría 'Distribución Geográfica' que contiene otros mapas
    if 'Distribución Geográfica' in categorias:
        del categorias['Distribución Geográfica']
    
    # Crear HTML
    html_content = f"""
    <!DOCTYPE html>
    <html lang="es">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Análisis de Datos de Seguridad y Criminalidad - Pantalla Completa</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1600px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f8f9fa;
            }}
            h1, h2, h3 {{
                color: #2c3e50;
            }}
            h1 {{
                border-bottom: 2px solid #3498db;
                padding-bottom: 10px;
                text-align: center;
            }}
            h2 {{
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            .header {{
                background-color: #3498db;
                color: white;
                padding: 20px;
                border-radius: 5px;
                margin-bottom: 30px;
                text-align: center;
            }}
            .section {{
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                padding: 20px;
                margin-bottom: 30px;
            }}
            .viz-container {{
                display: flex;
                flex-direction: column;
                gap: 30px;
            }}
            .viz-item {{
                width: 100%;
                margin-bottom: 30px;
                background-color: #f1f2f6;
                border-radius: 5px;
                overflow: hidden;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .viz-item h4 {{
                margin: 0;
                padding: 15px;
                background-color: #dfe4ea;
                font-size: 18px;
                text-align: center;
            }}
            iframe {{
                border: none;
                width: 100%;
                height: 800px;
            }}
            .footer {{
                text-align: center;
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #bdc3c7;
                color: #7f8c8d;
            }}
            .resumen {{
                padding: 15px;
                background-color: #e8f4f8;
                border-left: 4px solid #3498db;
                margin-bottom: 20px;
            }}
            .nav {{
                position: sticky;
                top: 0;
                background-color: #3498db;
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 20px;
                z-index: 100;
            }}
            .nav ul {{
                display: flex;
                list-style-type: none;
                padding: 0;
                margin: 0;
                justify-content: center;
                flex-wrap: wrap;
            }}
            .nav li {{
                margin: 0 15px;
            }}
            .nav a {{
                color: white;
                text-decoration: none;
                font-weight: bold;
            }}
            .nav a:hover {{
                text-decoration: underline;
            }}
            @media print {{
                .nav {{
                    display: none;
                }}
                iframe {{
                    height: 500px;
                }}
                .viz-item {{
                    page-break-inside: avoid;
                    margin-bottom: 50px;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Análisis de Datos de Seguridad y Criminalidad</h1>
            <h2>Visualizaciones a Pantalla Completa</h2>
            <p>Fecha de generación: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="nav">
            <ul>
                <li><a href="#intro">Introducción</a></li>
                {''.join([f'<li><a href="#{cat.lower().replace(" ", "-")}">{cat}</a></li>' for cat in categorias.keys()])}
                <li><a href="#conclusiones">Conclusiones</a></li>
            </ul>
        </div>
        
        <div id="intro" class="section">
            <h2>Introducción</h2>
            <p>Este informe presenta visualizaciones de diversos conjuntos de datos relacionados con la seguridad y la criminalidad.
            Las visualizaciones están organizadas por categorías y cada una ocupa el ancho completo de la página para un análisis más detallado.</p>
        </div>
    """
    
    # Agregar secciones para cada categoría
    for categoria, archivos in categorias.items():
        if archivos:
            seccion_id = categoria.lower().replace(" ", "-")
            html_content += f"""
            <div id="{seccion_id}" class="section">
                <h2>{categoria}</h2>
                <div class="viz-container">
            """
            
            for archivo in archivos:
                nombre_visual = archivo.replace('.html', '').replace('_', ' ').title()
                html_content += f"""
                    <div class="viz-item">
                        <h4>{nombre_visual}</h4>
                        <iframe src="../visualizaciones/{archivo}"></iframe>
                    </div>
                """
            
            html_content += """
                </div>
            </div>
            """
    
    # Conclusiones
    html_content += """
        <div id="conclusiones" class="section">
        <div class="footer">
            <p>Análisis realizado con Python utilizando pandas, matplotlib, seaborn, plotly y folium.</p>
            <p>Análisis de Datos de Seguridad y Criminalidad</p>
        </div>
    </body>
    </html>
    """
    
    # Guardar archivo HTML
    with open('informe/reporte_pantalla_completa.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Informe HTML a pantalla completa generado exitosamente en 'informe/reporte_pantalla_completa.html'")

# Generar informe HTML a pantalla completa
generar_informe_pantalla_completa()

# ==========================================
# Generar archivo README.md con instrucciones
# ==========================================

def generar_readme():
    print("\nGenerando archivo README.md con instrucciones...")
    
    readme_content = """# Análisis de Datos de Seguridad y Criminalidad

Este proyecto realiza un análisis exhaustivo de diversos conjuntos de datos relacionados con la seguridad y criminalidad. 

## Contenido

- **analisis_seguridad.py**: Analiza variables categóricas y tendencias temporales.
- **analisis_patrones.py**: Analiza patrones criminales y correlaciones.
- **analisis_geografico.py**: Realiza análisis geográficos y mapas.
- **fix_geographical_maps.py**: Genera mapas interactivos de delitos por departamentos y municipios.
- **analisis_principal.py**: Coordina la ejecución de todos los scripts de análisis.

## Resultados

Los resultados del análisis se encuentran en:

- **visualizaciones/**: Contiene todas las visualizaciones generadas.
- **informe/reporte_analisis.html**: Informe completo con todas las visualizaciones.

## Requisitos

Para ejecutar el análisis, se necesitan las siguientes bibliotecas:

```
pandas
numpy
matplotlib
seaborn
plotly
folium
```

Puede instalar todas las dependencias con:

```
pip install -r requirements.txt
```

## Ejecución

Para ejecutar el análisis completo:

```
python analisis_principal.py
```

"""
    
    # Guardar archivo README
    with open('README.md', 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print("Archivo README.md generado exitosamente")

# Generar README
generar_readme()

# Generar archivo de requisitos
def generar_requirements():
    print("\nGenerando archivo requirements.txt...")
    
    requirements = """pandas>=1.3.0
numpy>=1.20.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.0.0
folium>=0.14.0
"""
    
    # Guardar archivo de requisitos
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        f.write(requirements)
    
    print("Archivo requirements.txt generado exitosamente")

# Generar requirements.txt
generar_requirements()

print("\n" + "=" * 80)
print("ANÁLISIS COMPLETADO")
print("=" * 80)
print(f"Fecha y hora de finalización: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Visualizaciones generadas: {len(os.listdir('visualizaciones'))}")
print(f"Informes generados:")
print(f"  - Informe principal: informe/reporte_analisis.html")
print(f"  - Informe a pantalla completa: informe/reporte_pantalla_completa.html")
print("=" * 80)
print("\nPara ver los informes completos, abra los archivos HTML en su navegador.") 
print("El informe 'reporte_pantalla_completa.html' muestra cada visualización a tamaño completo para una mejor exploración de los detalles.") 