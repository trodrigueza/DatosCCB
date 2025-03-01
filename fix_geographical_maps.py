import pandas as pd
import numpy as np
import folium
from folium.plugins import MarkerCluster
import os
import warnings
warnings.filterwarnings('ignore')

# Crear directorio de visualizaciones si no existe
os.makedirs('visualizaciones', exist_ok=True)

print("Generando mapa de frentes de seguridad de Bogotá...")

# Diccionario de coordenadas de departamentos colombianos
# Formato: 'NOMBRE_DEPARTAMENTO': [latitud, longitud]
coordenadas_departamentos = {
    'ANTIOQUIA': [6.2476, -75.5658],
    'ATLÁNTICO': [10.9685, -74.7813],
    'BOGOTÁ D.C.': [4.6097, -74.0817],
    'BOLÍVAR': [10.3997, -75.5144],
    'BOYACÁ': [5.5544, -73.3575],
    'CALDAS': [5.0661, -75.5039],
    'CAQUETÁ': [1.6136, -75.6121],
    'CAUCA': [2.4448, -76.6142],
    'CESAR': [10.4631, -73.2532],
    'CÓRDOBA': [8.7489, -75.8800],
    'CUNDINAMARCA': [4.6019, -74.0819],
    'CHOCÓ': [5.6922, -76.6581],
    'HUILA': [2.5359, -75.5277],
    'LA GUAJIRA': [11.5444, -72.9072],
    'MAGDALENA': [11.2404, -74.1996],
    'META': [4.1429, -73.6259],
    'NARIÑO': [1.2136, -77.2811],
    'NORTE DE SANTANDER': [7.8939, -72.5078],
    'QUINDÍO': [4.5389, -75.6729],
    'RISARALDA': [4.8133, -75.6961],
    'SANTANDER': [7.1254, -73.1198],
    'SUCRE': [9.3048, -75.3975],
    'TOLIMA': [4.0993, -75.1538],
    'VALLE DEL CAUCA': [3.4516, -76.5320],
    'ARAUCA': [7.0907, -70.7616],
    'CASANARE': [5.3389, -72.3891],
    'PUTUMAYO': [0.4360, -76.5364],
    'AMAZONAS': [-1.4418, -71.5724],
    'GUAINÍA': [3.8608, -67.9249],
    'GUAVIARE': [2.5739, -72.6421],
    'VAUPÉS': [1.2537, -70.2337],
    'VICHADA': [4.4234, -69.2872],
    'SAN ANDRÉS': [12.5567, -81.7226],
    'PROVIDENCIA': [13.3498, -81.3757]
}

# Coordenadas de localidades en Bogotá
coordenadas_localidades = {
    'USAQUEN': [4.7110, -74.0324],
    'CHAPINERO': [4.6486, -74.0664],
    'SANTA FE': [4.5986, -74.0760],
    'SAN CRISTOBAL': [4.5721, -74.0887],
    'USME': [4.5271, -74.1141],
    'TUNJUELITO': [4.5724, -74.1384],
    'BOSA': [4.6300, -74.1984],
    'KENNEDY': [4.6510, -74.1617],
    'FONTIBON': [4.6709, -74.1469],
    'ENGATIVA': [4.7055, -74.1113],
    'SUBA': [4.7652, -74.0742],
    'BARRIOS UNIDOS': [4.6655, -74.0840],
    'TEUSAQUILLO': [4.6447, -74.0926],
    'MARTIRES': [4.6070, -74.0892],
    'ANTONIO NARIÑO': [4.5888, -74.1033],
    'PUENTE ARANDA': [4.6191, -74.1257],
    'CANDELARIA': [4.5960, -74.0743],
    'RAFAEL URIBE': [4.5683, -74.1193],
    'CIUDAD BOLIVAR': [4.5095, -74.1560],
    'SUMAPAZ': [4.2583, -74.2069]
}

# Coordenadas de algunas ciudades principales para demostraciones
coordenadas_municipios = {
    'BOGOTÁ D.C. (CT)': [4.6097, -74.0817],
    'MEDELLÍN': [6.2476, -75.5658],
    'CALI': [3.4516, -76.5320],
    'BARRANQUILLA': [10.9685, -74.7813],
    'CARTAGENA': [10.3997, -75.5144],
    'BUCARAMANGA': [7.1254, -73.1198],
    'PEREIRA': [4.8133, -75.6961],
    'SANTA MARTA': [11.2404, -74.1996],
    'CÚCUTA': [7.8939, -72.5078],
    'IBAGUÉ': [4.0993, -75.1538],
    'MANIZALES': [5.0661, -75.5039],
    'PASTO': [1.2136, -77.2811],
    'NEIVA': [2.5359, -75.5277],
    'VILLAVICENCIO': [4.1429, -73.6259],
    'ARMENIA': [4.5389, -75.6729],
    'POPAYÁN': [2.4448, -76.6142],
    'VALLEDUPAR': [10.4631, -73.2532],
    'MONTERÍA': [8.7489, -75.8800],
    'SINCELEJO': [9.3048, -75.3975],
    'TUNJA': [5.5544, -73.3575],
    'FLORENCIA': [1.6136, -75.6121],
    'RIOHACHA': [11.5444, -72.9072],
    'QUIBDÓ': [5.6922, -76.6581],
    'YOPAL': [5.3389, -72.3891],
    'MOCOA': [1.1477, -76.6481],
    'ARAUCA': [7.0907, -70.7616],
    'MITÚ': [1.2537, -70.2337],
    'PUERTO CARREÑO': [6.1852, -67.4931],
    'INÍRIDA': [3.8608, -67.9249],
    'LETICIA': [-4.2158, -69.9400],
    'SAN JOSÉ DEL GUAVIARE': [2.5739, -72.6421],
    'SAN ANDRÉS': [12.5567, -81.7226]
}

# Función para cargar y limpiar datasets
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

# Función para generar un mapa de frentes de seguridad de Bogotá por localidad
def generar_mapa_frentes_seguridad_bogota():
    """Genera un mapa que muestra los frentes de seguridad de Bogotá por localidad."""
    print("Generando mapa de frentes de seguridad de Bogotá por localidad...")
    
    # Cargar dataset de frentes de seguridad
    df_frentes = load_dataset("Frentes_De_Seguridad.csv")
    
    if df_frentes is None:
        print("  No se pudo cargar el dataset de Frentes de Seguridad")
        return False
    
    # Normalizar nombres de columnas
    new_cols = ['REGION', 'METROPOLITANA', 'DISTRITO', 'ESTACION', 'BARRIO', 'ZONA', 'NRO_INTEGRANTES', 'ESTADO']
    if len(df_frentes.columns) == 8:  # Si tiene las 8 columnas esperadas
        df_frentes.columns = new_cols
    
    # Filtrar solo los frentes de Bogotá
    if 'METROPOLITANA' in df_frentes.columns:
        df_bogota = df_frentes[df_frentes['METROPOLITANA'] == 'METROPOLITANA DE BOGOTA'].copy()
        print(f"  Frentes en Bogotá: {len(df_bogota)} de {len(df_frentes)} totales")
        
        # Extraer localidad de la última palabra de la columna ESTACION
        if 'ESTACION' in df_bogota.columns:
            # Extraer la última palabra de ESTACIÓN como localidad
            df_bogota['LOCALIDAD'] = df_bogota['ESTACION'].str.split().str[-1]
            
            # Crear un mapa base centrado en Bogotá
            mapa = folium.Map(
                location=[4.6097, -74.0817],  # Coordenadas de Bogotá
                zoom_start=11,
                tiles='CartoDB positron'
            )
            
            # Agregar un título al mapa
            titulo_html = '''
                <h3 align="center" style="font-size:16px"><b>Distribución de Frentes de Seguridad de Bogotá por Localidad</b></h3>
            '''
            mapa.get_root().html.add_child(folium.Element(titulo_html))
            
            # Agrupar frentes por localidad
            frentes_por_localidad = df_bogota.groupby('LOCALIDAD').size().reset_index(name='cantidad')
            
            # Normalizar nombres de localidades
            frentes_por_localidad['LOCALIDAD'] = frentes_por_localidad['LOCALIDAD'].str.upper()
            
            # Crear marcadores para cada localidad
            for _, row in frentes_por_localidad.iterrows():
                localidad = row['LOCALIDAD']
                cantidad = row['cantidad']
                
                # Buscar coordenadas de la localidad
                coords = None
                
                # Intentar encontrar en el diccionario de localidades
                if localidad in coordenadas_localidades:
                    coords = coordenadas_localidades[localidad]
                else:
                    # Usar coordenadas de Bogotá con un pequeño desplazamiento aleatorio
                    base_lat, base_lon = 4.6097, -74.0817
                    coords = [
                        base_lat + np.random.uniform(-0.05, 0.05),
                        base_lon + np.random.uniform(-0.05, 0.05)
                    ]
                
                # Crear círculo con tamaño proporcional a la cantidad de frentes
                radio = min(cantidad / 5, 20)  # Limitar tamaño máximo
                folium.Circle(
                    location=coords,
                    radius=radio * 100,  # Convertir a metros
                    color='blue',
                    fill=True,
                    fill_opacity=0.6,
                    popup=f"<b>{localidad}</b><br>Frentes de seguridad: {cantidad}"
                ).add_to(mapa)
                
                # Añadir etiqueta de texto
                folium.Marker(
                    location=coords,
                    icon=folium.DivIcon(
                        icon_size=(150, 36),
                        icon_anchor=(75, 0),
                        html=f'<div style="font-size: 10pt; text-align: center; background-color: rgba(255,255,255,0.7); border-radius: 3px; padding: 2px; width: 150px;">{localidad}<br>{cantidad} frentes</div>'
                    )
                ).add_to(mapa)
            
            # Guardar el mapa
            mapa.save('visualizaciones/Frentes_Seguridad_Bogota.html')
            print("  Mapa guardado como 'visualizaciones/Frentes_Seguridad_Bogota.html'")
            return True
        
        else:
            print("  No se encontró la columna ESTACION en el dataset de Frentes de Seguridad")
            return False
    else:
        print("  No se encontró la columna METROPOLITANA en el dataset de Frentes de Seguridad")
        return False

# Generar únicamente el mapa de frentes de seguridad de Bogotá
generar_mapa_frentes_seguridad_bogota()
print("Generación de mapas completada. Revise la carpeta 'visualizaciones'.") 