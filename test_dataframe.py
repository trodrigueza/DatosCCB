import pandas as pd

print("Cargando Hurto_Personas.csv...")
try:
    # Cargar directamente con coma como delimitador (que es lo que parece tener)
    df = pd.read_csv("datos/Hurto_Personas.csv", encoding='latin1', low_memory=False)
    
    print(f"Filas: {df.shape[0]}, Columnas: {df.shape[1]}")
    print("\nNombres de columnas:")
    for i, col in enumerate(df.columns):
        print(f"{i}: '{col}'")
    
    print("\nPrimeras 3 filas:")
    print(df.head(3))
    
except Exception as e:
    print(f"Error al cargar: {e}") 